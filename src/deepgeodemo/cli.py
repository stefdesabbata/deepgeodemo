import argparse
import importlib.util

from . import autoencoder_train_latent
from . import kmeans_sklearn
from . import kmeans_sklearn_search

def main():
    parser = argparse.ArgumentParser(description='DeepGeodemo Command-Line Interface (CLI).')
    parser.add_argument('-t', '--train_ae', action='store_true', help='Train the autoencoder.')
    parser.add_argument('-l', '--latent', action='store_true', help='Create latent representation.')
    parser.add_argument('-c', '--cluster', action='store_true', help='Run clustering using k-means.')
    parser.add_argument('-s', '--search', action='store_true', help='Run clustering in test mode to search for best k.')
    # parser.add_argument('-r', '--rapids', action='store_true', help='Use RAPIDS backend for clustering. Default is sklearn.')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('config', type=str, help='Path to the configuration YAML file.')

    args = parser.parse_args()

    # Run the appropriate module
    # Train autoencoder
    print("\nDeepGeodemo CLI")
    # print(f"{args.train_ae=} {args.latent=}")
    if args.train_ae and args.latent:
        print("Training autoencoder and creating latent representation.")
        autoencoder_train_latent.main(args.config, verbose=args.verbose)
    else:
        if args.train_ae:
            print("Training autoencoder.")
            autoencoder_train_latent.main(args.config, create_latent=False, verbose=args.verbose)
        # Create latent representation
        if args.latent:
            print("Creating latent representation.")
            autoencoder_train_latent.main(args.config, train_ae=False, verbose=args.verbose)
    
    # Clustering
    avail_cuml = importlib.util.find_spec("cuml")

    # Search mode
    if args.search:

        if avail_cuml is None:
            # sklearn backend (cpu)
            print("Running clustering in search mode to find best k, using scikit-learn backend.")
            kmeans_sklearn_search.main(args.config, args.verbose)
        else:
            # rapids backend
            # Importing libraries inside the function to avoid loading rapids if not needed
            from . import kmeans_rapids_search
            print("Running clustering in search mode to find best k, using RAPIDS backend.")
            kmeans_rapids_search.main(args.config, args.verbose)

    # Clustering mode
    if args.cluster:

        if avail_cuml is None:
            # sklearn backend (cpu)
            print("Clustering using k-means, using scikit-learn backend.")
            kmeans_sklearn.main(args.config, args.verbose)
        else:
            # rapids backend
            # Importing libraries inside the function to avoid loading rapids if not needed
            from . import kmeans_rapids
            print("Clustering using k-means, using RAPIDS backend.")
            kmeans_rapids.main(args.config, args.verbose)

    # No module selected, print help
    if not (args.train_ae or args.latent or args.search or args.cluster):
        parser.print_help()

if __name__ == '__main__':
    main()