# Configuration file

DeepGeoDemo is configured via a single YAML file, passed as the last argument to the CLI. A minimal configuration only needs to specify the input data, an identifier, and a few settings for the clustering set, while everything else would falls back to default values. The full set of supported keys is documented below and illustrated by the configurations in the [`example/` directory of the repository](https://github.com/stefdesabbata/deepgeodemo/tree/main/example).

The top-level structure includes:

- `data`: input dataset location and columns (see below).
- `working_dir`: where outputs, logs and plots are written. This is the directory where trained models, latent embeddings, cluster labels, logs and plots are written. It is created automatically if it does not exist. 
- `random_seed`: optional, for reproducibility. It seeds PyTorch for training and scikit-learn / cuML for clustering so that runs are reproducible.
- `autoencoder`: architecture, training and loss options (see below).
- `clustering`: k-means search and clustering options (see below).

## `data`

| Key | Type | Description |
| --- | --- | --- |
| `source` | string | Path to the input file. `.csv` and `.parquet` are supported. `~` is expanded to the user's home directory. |
| `nickname` | string | Short identifier for the dataset. Used in all output file names. |
| `id_col` | string | Name of the column that uniquely identifies each row. Carried through to the latent and cluster outputs. |
| `exclude_cols` | list of strings | Optional. Columns to carry through to the outputs but exclude from training and clustering (e.g. an existing classification label). |

## `autoencoder`

This section controls both the architecture of the autoencoder and how it is trained.

### Identifiers and outputs

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `nickname` | string |  | Short identifier for this autoencoder run, combined with the data nickname in output file names. |
| `version` | string |  | Version tag appended to the output file names (e.g. `0_1`). |
| `save_latent` | `csv` or `parquet` | `csv` | Format used to save the latent representation. |

### Training

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `max_epochs` | int | `100` | Maximum number of training epochs. |
| `batch_size` | int or float | full dataset | If `0 < batch_size <= 1`, interpreted as a fraction of the dataset. If `> 1`, interpreted as the exact number of rows. |
| `loader_workers` | int | `0` | Number of worker processes for the PyTorch `DataLoader`. |
| `learning_rate` | float | `1e-3` | Initial learning rate for the AdamW optimizer. |
| `patience` | int | `10` | Patience (in epochs) for the `ReduceLROnPlateau` scheduler that monitors the training loss. |
| `validate` | float |  | Optional fraction in `(0, 1)` of the dataset to use as a validation split. |
| `use_batch_norm` | bool | `false` | Add `BatchNorm1d` layers between hidden layers of the encoder and decoder. |
| `regu_weight_l2` | float | `0.0` | L2 weight decay applied through AdamW. |
| `regu_weight_l1` | float | `0.0` | L1 regularisation on all model weights. Only applied if `> 0`. |

### Architecture

The encoder layer sizes can be specified explicitly through `encoder.sizes`, or generated automatically from a target `depth` and `latent` size. The input layer size is inferred from the data (after dropping `id_col` and `exclude_cols`), so it should not be included in `sizes`. If `decoder.sizes` is omitted, the decoder defaults to the reverse of the encoder. The `LeakyReLU` activation is used after each layer, except for the last layer of both the encoder and the decoder, where the default is `Identity` but a different activation can be specified using `encoder.activation` and `decoder.activation`.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `depth` | int | `2` | Number of encoder layers. Used only when `encoder.sizes` is not given. |
| `latent` | int | `8` | Size of the latent layer. Used only when `encoder.sizes` is not given. |
| `encoder.sizes` | list of int |  | Explicit sizes of the encoder hidden and latent layers (excluding the input dimension). |
| `encoder.activation` | string | `Identity` | Final activation for the encoder. One of `Identity`, `JumpReLU`, `LeakyReLU`, `ReLU`, `Tanh`, `Sigmoid`. |
| `encoder.sparse` | mapping |  | Turns the encoder into a TopK sparse encoder. Simply including this key (even empty) enables sparsity. |
| `encoder.sparse.topk_k` | int | half of latent (when sparse is active) | Number of active (top-k) latent neurons per sample. |
| `encoder.sparse.use_batch_norm` | bool | `false` | Add a `BatchNorm1d` layer before the TopK activation. |
| `decoder.sizes` | list of int | reverse of encoder | Explicit sizes of the decoder hidden layers (the output dimension is inferred from the data and appended automatically). |
| `decoder.activation` | string | `Identity` | Final activation for the decoder. Same options as `encoder.activation`. |

### Loss weights

All loss weights default to `0.0` and are only added to the total loss when set above zero. The reconstruction loss (normalised MSE) is always applied.

| Key | Type | Description |
| --- | --- | --- |
| `loss_weights.latent_l1` | float | Weight for an L1 penalty on the latent embeddings. |
| `loss_weights.latent_l0` | float | Weight for an L0 penalty on the latent embeddings. |
| `loss_weights.covariance` | float | Weight for a covariance penalty that discourages correlated latent dimensions. |
| `loss_weights.auxk` | float | Weight for the auxiliary TopK loss that mitigates dead neurons. Only used when `encoder.sparse` is set. |

## `clustering`

K-means clustering runs on the latent embeddings produced by the autoencoder. Two modes are supported: **search** (`-s`), which tries a range of values of k and writes diagnostic plots, and **cluster** (`-c`), which fits and saves one or more chosen values of k. A single configuration file can declare both.

| Key | Type | Description |
| --- | --- | --- |
| `nickname` | string | Short identifier for this clustering run, used in output file and column names. |
| `version` | string | Version tag for the clustering run. |
| `test.from` | int | Smallest k to try in search mode. |
| `test.to` | int | Largest k to try in search mode (inclusive). |
| `test.n_init` | int | `n_init` passed to `KMeans` during the search. |
| `test.max_iter` | int | `max_iter` passed to `KMeans` during the search. |
| `cluster.k` | list of int | Values of k to fit in cluster mode. Each produces a column in the output table. |
| `cluster.n_init` | int | `n_init` for the final clustering. |
| `cluster.max_iter` | int | `max_iter` for the final clustering. |
| `cluster.save_clusters` | `csv` or `parquet` | Format used to save the cluster labels. Defaults to `csv`. |

Search mode writes a clustergram, a WCSS (elbow) plot and a silhouette score plot into `working_dir`; cluster mode writes one column of labels per value of k in `cluster.k`.

## Example configurations

The [`example/` directory of the repository](https://github.com/stefdesabbata/deepgeodemo/tree/main/example) includes several ready-to-run configurations that exercise the options above:

- `example_minimal.yml`: smallest valid configuration, relying on the default `depth` and `latent` size.
- `example.yml`: explicit encoder and decoder sizes.
- `example_depth_latent.yml`: auto-generated layer sizes from `depth` and `latent`.
- `example_with_validate.yml`: enables a validation split.
- `example_sparse_relu.yml`: TopK sparse encoder with a ReLU base activation and latent L0 / auxk penalties.
- `example_sparse_jumprelu.yml`: same as above, using JumpReLU as the base activation.
