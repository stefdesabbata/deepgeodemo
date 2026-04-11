# Libraries
import os
import argparse
import random
import yaml
import pandas as pd
import cudf
from cuml.cluster import KMeans


def cluster(geodemo_config, verbose):

    # Configuration -----------------------------------------------------------

    if 'random_seed' in geodemo_config:
        random_seed       = geodemo_config['random_seed']
    else:
        random_seed       = random.randint(0, 999999)
        print(f'Missing random seed set to: {random_seed=}') if verbose else None

    dir_project           = os.path.expanduser(
                            geodemo_config['working_dir'])

    dataset_nickname      = geodemo_config['data']['nickname']
    dataset_id_col        = geodemo_config['data']['id_col']
    if 'exclude_cols' in    geodemo_config['data']:
        dataset_excl_cols = geodemo_config['data']['exclude_cols']
    else:
        dataset_excl_cols = []

    model_nickname        = geodemo_config['autoencoder']['nickname']
    model_version         = geodemo_config['autoencoder']['version']

    clust_nickname        = geodemo_config['clustering']['nickname']
    clust_version         = geodemo_config['clustering']['version']

    clust_k_selectd   = geodemo_config['clustering']['cluster']['k']
    clust_n_init      = geodemo_config['clustering']['cluster']['n_init']
    clust_max_iter    = geodemo_config['clustering']['cluster']['max_iter']
    clust_save_clusts = geodemo_config['clustering']['cluster']['save_clusters'] if 'save_clusters' in geodemo_config['clustering']['cluster'] else None


    # Files and directories -----------------------------------------------

    latent_csv_path  = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__latent.csv')
    latent_prq_path  = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__latent.parquet')

    if os.path.exists(latent_prq_path):
        latent = pd.read_parquet(latent_prq_path)
    elif os.path.exists(latent_csv_path):
        latent = pd.read_csv(latent_csv_path)
    else:
        raise FileNotFoundError(f"Could not find {latent_csv_path} or {latent_prq_path}")

    clust_output_csv = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__kmeans_clusters.csv')
    clust_output_prq = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__kmeans_clusters.parquet')
    

    # Preparation -------------------------------------------------------------
    
    ids           = latent[dataset_id_col]
    latent        = latent.drop(columns=[dataset_id_col])
    if len(dataset_excl_cols) > 0:
        excl_cols = latent[dataset_excl_cols]
        latent    = latent.drop(columns=dataset_excl_cols)

    # weights    = pd.read_parquet(weights_path)
    # assert latent.shape[0]   == weights.shape[0],   'Data and weights have different number of rows'
    # assert latent[dataset_id_col] == weights[dataset_id_col], 'Data and weights have different ids'
    # weights    = weights.drop(columns=[dataset_id_col])

    # Create cuml dataframes
    latent_cudf    = cudf.DataFrame(latent)
    # weights_cudf = cudf.DataFrame(weights)


    # Clustering --------------------------------------------------------------

    clusters_for_k = {}
    for k in clust_k_selectd:
        print(f'Fitting k={k}') if verbose else None
        kmeans = KMeans(n_clusters=k, max_iter=clust_max_iter, n_init=clust_n_init, verbose=verbose, random_state=random_seed).fit(latent_cudf)
        clusters_for_k[f'{clust_nickname}_{clust_version}_k{k:03d}_now'] = kmeans.labels_.to_numpy()
        del kmeans


    # Save clusters -----------------------------------------------------------

    print('Saving clusters') if verbose else None
    clusters_for_k_df = pd.DataFrame(clusters_for_k)
    parts = [ids]
    if len(dataset_excl_cols) > 0:
        parts.append(excl_cols)
    parts.append(clusters_for_k_df)

    if clust_save_clusts == "csv":
        pd.concat(parts, axis=1).to_csv(clust_output_csv, index=False)
    elif clust_save_clusts == "parquet":
        pd.concat(parts, axis=1).to_parquet(clust_output_prq, index=False)
    else:
        print('No save format specified, saving as CSV') if verbose else None
        pd.concat(parts, axis=1).to_csv(clust_output_csv, index=False)



# Main --------------------------------------------------------------------

def main(config, verbose):

    # Configuration -----------------------------------------------------------

    with open(config, 'r') as file:
        geodemo_config = yaml.safe_load(file)
    print(f'{geodemo_config=}') if verbose else None

    # Clustering --------------------------------------------------------------

    cluster(geodemo_config, verbose)



if __name__ == "__main__":
     
    # Parse arguments --------------------------------------------------------

    parser = argparse.ArgumentParser(description='Clustering.')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('config', type=str, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    main(args.config, args.verbose)
