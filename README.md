# DeepGeoDemo: Deep Embedding Geodemographics Made Simple

## Overview

[DeepGeoDemo](https://deepgeodemo.readthedocs.io/en/latest/) makes it easy to build geodemographic classifications powered by deep embeddings. Configure your autoencoder in a simple YAML file and run the full pipeline from the command line, or import DeepGeoDemo as a Python library when you need more control.



## Installation

The package requires Python 3.12. The package can be installed using the following commands. This will install the package with the CPU dependencies, i.e., the cpu version of [PyTorch](https://pytorch.org/) and [scikit-learn](https://scikit-learn.org/stable/index.html) for clustering.

Create a virtual environment, e.g. using conda.

```bash
conda create -n deepgeodemo python=3.12
conda activate deepgeodemo
```

Install the [deepgeodemo package](https://pypi.org/project/deepgeodemo/) via `pip`.

```bash
pip install deepgeodemo
```

Alternatively, the package can be installed with the GPU dependencies if available and [RAPIDS](https://docs.rapids.ai/) for the clustering.

```bash
pip install --extra-index-url=https://pypi.nvidia.com deepgeodemo[gpu]
```


## Usage

The commands below illustrate how to use the Command-Line Interface (CLI) to run the tool using the example configuration file `example/example.yml` and data (eight random blobs in sixteen dimensions). The `-t` flag is used to train the autoencoder and `l` to create the latent representation. The `-s` flag is used to search for the best k. The `-c` flag is used to run clustering using k-means. If available, the cuml backend is used for clustering. The `-v` flag is optional and is used to display the progress of the process. 

Train the autoencoder.

```bash
deepgeodemo -tv example/example.yml
```

Create latent representation using on the previously trained autoencoder. Note that this will load a model from disk, and thus it will rais a warning message, as that can result in **arbitrary code execution**. Do it only if you got the file from a **trusted** source -- e.g., a model file you trained yourself, using the command above.

```bash
deepgeodemo -lv example/example.yml
```

Alternatively, you can train the autoencoder and create the latent representation in one go. In this case, the autoencoder will still be saved, but the latent representation will be created directly with the model in memory (rather than loading from the disk).

```bash
deepgeodemo -tlv example/example.yml
```

Run clustering in test mode to search for best k. Add `-r` for RAPIDS backend, default is scikit-learn.

```bash
deepgeodemo -sv example/example.yml
```

Run clustering using k-means. Add `-r` for RAPIDS backend, default is scikit-learn.

```bash
deepgeodemo -cv example/example.yml
```

Alternatively, you can run everything in one go as well.

```bash
deepgeodemo -tlscv example/example.yml
```

For a more concrete example, you can test the tool using the 2021 OAC data available from [Jakub Wyszomierski's repo](https://github.com/jakubwyszomierski/OAC2021-2). Download the [Clean data](https://liveuclac-my.sharepoint.com/:f:/g/personal/zcfajwy_ucl_ac_uk/Eqd1EV2WgOFJmZ7kLx-oDYMBdxqNe9IJmli6M8S-e91F0g?e=M9wh5j) used to create the [2021 OAC](https://data.cdrc.ac.uk/dataset/output-area-classification-2021), unzip the file and set the value of `data: source` to the location of one of the file datasets on your computer. It is advisable to normalise the data before training the autoencoder, e.g., using min-max scaling. Please increase the number of epochs and the number of clustering iteration to get meaninful results.



## Configuration file

DeepGeoDemo is configured via a single YAML file, passed as the last argument to the CLI. A minimal configuration only needs to specify the input data, an identifier, and a few settings for the clustering set, while everything else would falls back to default values. The full set of supported keys is documented below and illustrated by the configurations in [example/](example/).

The top-level structure includes:

- `data`: input dataset location and columns (see below).
- `working_dir`: where outputs, logs and plots are written. This is the directory where trained models, latent embeddings, cluster labels, logs and plots are written. It is created automatically if it does not exist. 
- `random_seed`: optional, for reproducibility. It seeds PyTorch for training and scikit-learn / cuML for clustering so that runs are reproducible.
- `autoencoder`: architecture, training and loss options (see below).
- `clustering`: k-means search and clustering options (see below).

### `data`

| Key | Type | Description |
| --- | --- | --- |
| `source` | string | Path to the input file. `.csv` and `.parquet` are supported. `~` is expanded to the user's home directory. |
| `nickname` | string | Short identifier for the dataset. Used in all output file names. |
| `id_col` | string | Name of the column that uniquely identifies each row. Carried through to the latent and cluster outputs. |
| `exclude_cols` | list of strings | Optional. Columns to carry through to the outputs but exclude from training and clustering (e.g. an existing classification label). |

### `autoencoder`

This section controls both the architecture of the autoencoder and how it is trained.

**Identifiers and outputs**

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `nickname` | string |  | Short identifier for this autoencoder run, combined with the data nickname in output file names. |
| `version` | string |  | Version tag appended to the output file names (e.g. `0_1`). |
| `save_latent` | `csv` or `parquet` | `csv` | Format used to save the latent representation. |

**Training**

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

**Architecture**

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

**Loss weights**

All loss weights default to `0.0` and are only added to the total loss when set above zero. The reconstruction loss (normalised MSE) is always applied.

| Key | Type | Description |
| --- | --- | --- |
| `loss_weights.latent_l1` | float | Weight for an L1 penalty on the latent embeddings. |
| `loss_weights.latent_l0` | float | Weight for an L0 penalty on the latent embeddings. |
| `loss_weights.covariance` | float | Weight for a covariance penalty that discourages correlated latent dimensions. |
| `loss_weights.auxk` | float | Weight for the auxiliary TopK loss that mitigates dead neurons. Only used when `encoder.sparse` is set. |

### `clustering`

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

### Example configurations

The [example/](example/) directory includes several ready-to-run configurations that exercise the options above:

- [example_minimal.yml](example/example_minimal.yml): smallest valid configuration, relying on the default `depth` and `latent` size.
- [example.yml](example/example.yml): explicit encoder and decoder sizes.
- [example_depth_latent.yml](example/example_depth_latent.yml): auto-generated layer sizes from `depth` and `latent`.
- [example_with_validate.yml](example/example_with_validate.yml): enables a validation split.
- [example_sparse_relu.yml](example/example_sparse_relu.yml): TopK sparse encoder with a ReLU base activation and latent L0 / auxk penalties.
- [example_sparse_jumprelu.yml](example/example_sparse_jumprelu.yml): same as above, using JumpReLU as the base activation.



## Citation

If you would like to cite this approach in your work, please reference the following paper (author accepted manuscript version available in the [papers/](papers/) folder):

```bibtex
@inproceedings{desabbata2019deepgeodemo,
  author    = {De Sabbata, Stef and Liu, Pengyuan},
  title     = {Deep learning geodemographics with autoencoders and geographic convolution},
  booktitle = {Accepted Short Papers and Posters from the 22nd {AGILE} Conference on Geo-information Science},
  editor    = {Kyriakidis, Phaedon and Hadjimitsis, Diofantos and Skarlatos, Dimitrios and Mansourian, Ali},
  year      = {2019},
  month     = {June},
  address   = {Limassol, Cyprus},
  publisher = {Stichting AGILE},
  isbn      = {978-90-816960-9-8}
}
```


## Acknowledgement

Many thanks to [Owen Goodwin](https://github.com/ogoodwin505), [Pengyuan Liu](https://github.com/PengyuanLiu1993) and [Alex Singleton](https://github.com/alexsingleton) for their collaboration on this project and for testing the pre-alpha versions of the tool.
