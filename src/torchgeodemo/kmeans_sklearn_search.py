# Libraries
import os
import argparse
from random import random
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score
from clustergram import Clustergram


def main(config, verbose):

    # Configuration -----------------------------------------------------------

    with open(config, 'r') as file:
        geodemo_config = yaml.safe_load(file)
    print(f'{geodemo_config=}') if verbose else None

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

    clust_k_to_test   = range(
                         geodemo_config['clustering']['test']['from'], 
                        (geodemo_config['clustering']['test']['to'] + 1))
    clust_n_init      =  geodemo_config['clustering']['test']['n_init']
    clust_max_iter    =  geodemo_config['clustering']['test']['max_iter']
    

    # Files and directories -----------------------------------------------

    latent_csv_path  = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__latent.csv')
    latent_prq_path  = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__latent.parquet')

    if os.path.exists(latent_prq_path):
        latent = pd.read_parquet(latent_prq_path)
    elif os.path.exists(latent_csv_path):
        latent = pd.read_csv(latent_csv_path)
    else:
        raise FileNotFoundError(f"Could not find {latent_csv_path} or {latent_prq_path}")
    
    plot_cgrm_path   = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__kmeans_cgrm.png')
    plot_wcss_path   = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__kmeans_wcss.png')
    plot_silh_path   = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__kmeans_silh.png')

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


    # Calculate and plot scores -----------------------------------------------

    # Fit clustergram
    cgram = Clustergram(
        clust_k_to_test, 
        backend=None, 
        max_iter=clust_max_iter,
        n_init=clust_n_init,
        verbose=verbose)
    
    print('Fitting clustergram') if verbose else None
    cgram.fit(latent) #, sample_weight=weights_cudf)

    # Calculate WCSS and silhouette scores
    wcss_scores = []
    silh_scores = []

    for k in clust_k_to_test:
        print(f'Calculating scores for k={k}') if verbose else None
        kmeans = KMeans(n_clusters=k, max_iter=clust_max_iter, n_init=clust_n_init, verbose=verbose, random_state=random_seed).fit(latent) #, sample_weight=weights_cudf)
        wcss_scores.append(kmeans.inertia_)
        print('Calculating silhouette score') if verbose else None
        silh_scores.append(silhouette_score(latent, kmeans.labels_))

    # Plot clustergram, WCSS and silhouette scores

    print('Plotting') if verbose else None
    cgram.plot(figsize=(16, 8))
    plt.savefig(plot_cgrm_path)
    plt.close()

    plt.figure(figsize=(16,8))
    sns.lineplot(x=clust_k_to_test, y=wcss_scores, marker='o', linestyle='-')
    plt.xlabel('k')
    plt.ylabel('WCSS')
    plt.xticks(clust_k_to_test)
    plt.savefig(plot_wcss_path, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(16,8))
    sns.lineplot(x=clust_k_to_test, y=silh_scores, marker='o', linestyle='-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.xticks(clust_k_to_test)
    plt.savefig(plot_silh_path, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
     
    # Parse arguments --------------------------------------------------------

    parser = argparse.ArgumentParser(description='Clustering.')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('config', type=str, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    main(args.config, args.verbose)
