# Libraries
import sys
import os
import gc
import argparse
import yaml
import pprint
import numpy as np
import pandas as pd
import math
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger


# This project's modules
from .models import AutoEncoder

# Utils -------------------------------------------------------------------

# Generate autoencoder sizes if none are provided
def generate_ae_sizes(input_size, latent_size, depth):
    if depth < 1:
        depth = 1
    return list(reversed([math.ceil(i) for i in np.linspace(latent_size, input_size, depth+1)]))


# Train and/or create latent ----------------------------------------------

def train_latent(geodemo_config, train_ae=True, create_latent=True, verbose=True):

    # Configuration -------------------------------------------------------
    report = None
    pprint.pprint(geodemo_config) if verbose else None
    geodemo_config_pp = pprint.pformat(geodemo_config)

    if 'random_seed' in geodemo_config:
        torch.manual_seed(geodemo_config['random_seed'])

    dataset_path      = os.path.expanduser(
                        geodemo_config['data']['source'])
    dataset_nickname  = geodemo_config['data']['nickname']

    model_nickname    = geodemo_config['autoencoder']['nickname']
    model_version     = geodemo_config['autoencoder']['version']
    model_save_latent = geodemo_config['autoencoder']['save_latent'] if 'save_latent' in geodemo_config['autoencoder'] else None
    
    # Files and directories -----------------------------------------------

    dir_project       = os.path.expanduser(
                        geodemo_config['working_dir'])
    dir_logs          = os.path.join(dir_project, 'logs')

    if not os.path.exists(dir_project):
        os.makedirs(dir_project)

    latent_csv_path  = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__latent.csv')
    latent_prq_path  = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__latent.parquet')
    model_path       = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__model.pth')
    model_info_path  = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__model__info.txt')


    # Load data -----------------------------------------------------------

    if dataset_path.endswith('.csv'):
        input_dataset = pd.read_csv(dataset_path)
    elif dataset_path.endswith('.parquet'):
        input_dataset = pd.read_parquet(dataset_path)
    else:
        raise ValueError('Unsupported file format. Only CSV and Parquet are currently supported.')

    # Extract ID
    dataset_id_col        = geodemo_config['data']['id_col']
    ids                   = input_dataset[dataset_id_col]
    input_dataset         = input_dataset.drop(columns=[dataset_id_col])
    model_str             = f'\n\nInput dataset shape: {input_dataset.shape}\n'
    model_str            += str(input_dataset.describe()) + '\n\n'
    # Extract excluded columns
    if 'exclude_cols' in    geodemo_config['data']:
        dataset_excl_cols = geodemo_config['data']['exclude_cols']
    else:
        dataset_excl_cols = []
    if len(dataset_excl_cols) > 0:
        excl_cols         = input_dataset[dataset_excl_cols]
        input_dataset     = input_dataset.drop(columns=dataset_excl_cols)

    # Create tensor and data loader
    data_tensor      = torch.tensor(input_dataset.values).float()
    data_tensor_nrow = data_tensor.shape[0]
    data_tensor_ncol = data_tensor.shape[1]


    if train_ae:

        # Model ---------------------------------------------------------------

        # Training parameters
        model_max_epochs      = geodemo_config['autoencoder']['max_epochs'] if 'max_epochs' in geodemo_config['autoencoder'] else 100
        model_batch_size      = geodemo_config['autoencoder']['batch_size'] if 'batch_size' in geodemo_config['autoencoder'] else -1
        model_loader_workers  = geodemo_config['autoencoder']['loader_workers'] if 'loader_workers' in geodemo_config['autoencoder'] else 0
        # Set batch size
        if model_batch_size   <= 0.0:
            data_batch_size   = data_tensor_nrow
        elif model_batch_size <= 1.0:
            data_batch_size   = math.ceil(data_tensor_nrow*model_batch_size)
        elif model_batch_size <= data_tensor_nrow:
            data_batch_size   = model_batch_size
        else:
            data_batch_size   = data_tensor_nrow

        # Autoencoder parameters
        ae_args            = {}
        ae_args['verbose'] = verbose

        if 'use_batch_norm' in geodemo_config['autoencoder']:
            ae_args['use_batch_norm'] = geodemo_config['autoencoder']['use_batch_norm']
        if 'regu_weight_l2' in geodemo_config['autoencoder']:
            ae_args['regu_weight_l2'] = geodemo_config['autoencoder']['regu_weight_l2']
        if 'regu_weight_l1' in geodemo_config['autoencoder']:
            ae_args['regu_weight_l1'] = geodemo_config['autoencoder']['regu_weight_l1']
        if 'learning_rate' in geodemo_config['autoencoder']:
            ae_args['learning_rate'] = geodemo_config['autoencoder']['learning_rate']
        if 'patience' in geodemo_config['autoencoder']:
            ae_args['patience'] = geodemo_config['autoencoder']['patience']
        # if 'pretrain' in geodemo_config['autoencoder']:
        #     if 'epochs' in geodemo_config['autoencoder']['pretrain']:
        #         ae_args['pretrain_epochs'] = geodemo_config['autoencoder']['pretrain']['epochs']
        #     if 'learning_rate' in geodemo_config['autoencoder']['pretrain']:
        #         ae_args['pretrain_learning_rate'] = geodemo_config['autoencoder']['pretrain']['learning_rate']

        # Validation
        if 'validate' in geodemo_config['autoencoder']:
            ae_validate          = True
            ae_validat_split     = geodemo_config['autoencoder']['validate']
            if (not isinstance(ae_validat_split, float)) or ae_validat_split <= 0.0 or ae_validat_split >= 1.0:
                ae_validate      = False
                ae_validat_split = 0.0
                print('No validation.', flush=True)
        else:
            ae_validate          = False
            ae_validat_split     = 0.0
            print('No validation.', flush=True)

        # Encoder sizes
        if 'encoder' in geodemo_config['autoencoder']:
            ae_encoder_sizes  = geodemo_config['autoencoder']['encoder']['sizes']
            ae_encoder_sizes  = [data_tensor_ncol] + ae_encoder_sizes
        else:
            if 'depth' in       geodemo_config['autoencoder']:
                ae_depth      = geodemo_config['autoencoder']['depth']
            else:
                ae_depth      = 2
                print(f'No encoder sizes or depth specified. Using default depth of {ae_depth}.', flush=True)
            if 'latent' in      geodemo_config['autoencoder']:
                ae_latent     = geodemo_config['autoencoder']['latent']
            else:
                ae_latent     = 8
                print(f'No latent size specified. Using default latent size of {ae_latent}.', flush=True)
            ae_encoder_sizes  = generate_ae_sizes(data_tensor_ncol, ae_latent, ae_depth)
            print(f'Using computed sizes: {ae_encoder_sizes}', flush=True)

        # Other autoencoder parameters
        if 'encoder' in                                 geodemo_config['autoencoder']:
            if 'activation' in                          geodemo_config['autoencoder']['encoder']:
                ae_args['encoder_activation']         = geodemo_config['autoencoder']['encoder']['activation']
            if 'loss_weights' in                        geodemo_config['autoencoder']['encoder']:
                if 'latent_l0' in                       geodemo_config['autoencoder']['encoder']['loss_weights']:
                    ae_args['loss_weight_latent_l0']  = geodemo_config['autoencoder']['encoder']['loss_weights']['latent_l0']
                if 'latent_l1' in                       geodemo_config['autoencoder']['encoder']['loss_weights']:
                    ae_args['loss_weight_latent_l1']  = geodemo_config['autoencoder']['encoder']['loss_weights']['latent_l1']
                if 'covariance' in                      geodemo_config['autoencoder']['encoder']['loss_weights']:
                    ae_args['loss_weight_covariance'] = geodemo_config['autoencoder']['encoder']['loss_weights']['covariance']
                if 'auxk' in                            geodemo_config['autoencoder']['encoder']['loss_weights']:
                    ae_args['loss_weight_auxk']       = geodemo_config['autoencoder']['encoder']['loss_weights']['auxk']
            if 'sparse' in                              geodemo_config['autoencoder']['encoder']:
                ae_args['encoder_sparse']             = True
                ae_args['encoder_sparse_topk_k']      = geodemo_config['autoencoder']['encoder']['sparse']['topk_k']

        # if 'dcc' in                                     geodemo_config['autoencoder']:
        #     ae_args['dcc_from_epoch']                 = geodemo_config['autoencoder']['dcc']['from_epoch']
        #     if 'neighbours' in                          geodemo_config['autoencoder']['dcc']:
        #         ae_args['dcc_neighbours']             = geodemo_config['autoencoder']['dcc']['neighbours']
        #     if 'nn_lambda' in                           geodemo_config['autoencoder']['dcc']:
        #         ae_args['dcc_nn_lambda']              = geodemo_config['autoencoder']['dcc']['nn_lambda']

        if 'decoder' in                                 geodemo_config['autoencoder']:
            if 'sizes' in                               geodemo_config['autoencoder']['decoder']:
                ae_args['decoder_sizes']              = geodemo_config['autoencoder']['decoder']['sizes']
                ae_args['decoder_sizes']              = ae_args['decoder_sizes'] + [data_tensor_ncol]
            if 'activation' in                          geodemo_config['autoencoder']['decoder']:
                ae_args['decoder_activation']         = geodemo_config['autoencoder']['decoder']['activation']

        print(f'\n{ae_args=}\n', flush=True) if verbose else None


        # Training ----------------------------------------------------------------

        # Create train and (optional) validation loaders
        if ae_validate:
            split_size_val   = math.floor(data_tensor_nrow * ae_validat_split)
            split_size_train = data_tensor_nrow - split_size_val
            train_dataset, val_dataset = torch.utils.data.random_split(
                data_tensor,
                [split_size_train, split_size_val]
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=data_batch_size,
                shuffle=False,
                num_workers=model_loader_workers
                )
        else:
            train_dataset = data_tensor
            val_loader = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=data_batch_size,
            shuffle=True,
            num_workers=model_loader_workers
            )

        # Create model
        geodemo_ae = AutoEncoder(encoder_sizes = ae_encoder_sizes, **ae_args)
        # Save model info
        model_str += str(geodemo_ae)
        model_str += f'\nuse_batch_norm = {geodemo_ae.use_batch_norm}'
        model_str += f'\nloss_weight_latent_l1 = {geodemo_ae.loss_weight_latent_l1}'
        model_str += f'\nloss_weight_covariance = {geodemo_ae.loss_weight_covariance}'
        model_str += f'\nregu_weight_l2 = {geodemo_ae.regu_weight_l2}'
        model_str += f'\nregu_weight_l1 = {geodemo_ae.regu_weight_l1}'
        model_str += f'\nlearning_rate = {geodemo_ae.learning_rate}'
        # model_str += f'\npretrain_epochs = {geodemo_ae.pretrain_epochs}' if geodemo_ae.pretrain_epochs > 0 else ''
        # model_str += f'\npretrain_learning_rate = {geodemo_ae.pretrain_learning_rate}' if geodemo_ae.pretrain_learning_rate > 0 else ''
        model_str += f'\n\n'
        print(model_str, flush=True) if verbose else None
        with open(model_info_path, 'w') as f:
            f.write('Config:\n')
            f.write(geodemo_config_pp)
            f.write('\n\n')
            f.write(model_str)
            f.write('\n\n')

        # Create loggers
        logger_test_name = f'log_{dataset_nickname}_{model_nickname}_v{model_version}'
        logger_folder = dir_logs
        logger_tb = TensorBoardLogger(logger_folder, name=logger_test_name)
        logger_csv = CSVLogger(logger_folder, name=logger_test_name)

        # Train the model
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        trainer = Trainer(
            devices=1, 
            accelerator=accelerator,
            logger=[logger_tb,logger_csv],
            callbacks=[lr_monitor], 
            max_epochs=model_max_epochs,
            enable_progress_bar=verbose
            )
        if ae_validate:
            trainer.fit(
                model=geodemo_ae,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader
                )
        else:
            trainer.fit(
                model=geodemo_ae,
                train_dataloaders=train_loader
                )

        # Print final metrics
        trainer_final_metrics = trainer.logged_metrics
        report = trainer_final_metrics
        trainer_final_metrics_str = 'Final metrics:\n'
        for key, value in trainer_final_metrics.items():
            trainer_final_metrics_str += f'{key}: {value}\n'
        trainer_final_metrics_str += '\n'
        print(trainer_final_metrics_str, flush=True) if verbose else None
        with open(model_info_path, 'a') as f:
            f.write(trainer_final_metrics_str)

        # Save model
        torch.save(geodemo_ae, model_path)

        # Release DataLoader workers
        print('Releasing DataLoader workers...', flush=True) if verbose else None
        del train_loader
        del val_loader
        gc.collect()

    # Load model and create latent ----------------------------------------
    else:

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'The model file {model_path} does not exist.')

        # Load trained model
        print('\n>>> WARNING <<<\nLoading model from disk.\nThis can result in **arbitrary code execution**. Do it only if you got the file from a **trusted** source!')
        response = input('Do you want to continue loading the model? (y/N): ')
        if response.lower() != 'y':
            print('Aborting operation.')
            sys.exit(0)
        geodemo_ae = torch.load(model_path, weights_only=False)


    # Create latent -------------------------------------------------------
    if create_latent:

        for f in [latent_csv_path, latent_prq_path]:
            if os.path.exists(f):
                raise FileExistsError(f'The file {f} already exists.')

        # Generate latent in batches and combine results
        with torch.no_grad():
            geodemo_ae.eval()

            latent_batches = []
                
            # Create tensor and data loader
            data_tensor = torch.tensor(input_dataset.values).float()
            data_tensor_nrow = data_tensor.shape[0]
            data_loader_for_encoder = torch.utils.data.DataLoader(
                data_tensor, 
                batch_size=math.ceil(data_tensor_nrow*0.01),
                shuffle=False # IMPORTANT: must be False to match the ids
                )
            for batch in data_loader_for_encoder:
                    latent_batch = geodemo_ae.encode(batch).cpu().detach().numpy()
                    latent_batches.append(latent_batch)
            latent = np.vstack(latent_batches)

            # Write reduced_data to a parquet file in the OAC folder
            latent_code_digits = int(np.ceil(np.log10(latent.shape[1])))
            latent_df = pd.DataFrame(latent)
            latent_df = latent_df.rename(columns=lambda x: 'EMB_'+'{:0{digits}d}'.format(x, digits=latent_code_digits))
            # Create output df
            if len(dataset_excl_cols) > 0:
                output_df = pd.concat([ids, excl_cols, latent_df], axis=1)
            else:
                output_df = pd.concat([ids, latent_df], axis=1)
            # Save output df
            if model_save_latent == 'csv':
                output_df.to_csv(latent_csv_path, index=False)
            elif model_save_latent == 'parquet':
                output_df.to_parquet(latent_prq_path, index=False)
            else:
                print(f'No dataset output format specified. Saving as CSV by default.') if verbose else None
                output_df.to_csv(latent_csv_path, index=False)
    
    return report


# Main --------------------------------------------------------------------

def main(config, train_ae=True, create_latent=True, verbose=True):

    # Configuration -------------------------------------------------------

    with open(config, 'r') as file:
        geodemo_config = yaml.safe_load(file)
    print(f'\n{geodemo_config=}\n', flush=True) if verbose else None

    report = train_latent(geodemo_config, train_ae, create_latent, verbose)
    return report



if __name__ == '__main__':
     
    # Parse arguments --------------------------------------------------------

    parser = argparse.ArgumentParser(description='Train AutoEncoder model.')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('config', type=str, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    main(args.config, verbose=args.verbose)