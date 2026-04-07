# TorchGeoDemo: Deep Embedding Geodemographics Made Simple

## Overview

TorchGeoDemo makes it easy to build geodemographic classifications powered by deep embeddings. Configure your autoencoder in a simple YAML file and run the full pipeline from the command line, or import TorchGeoDemo as a Python library when you need more control.



## Installation

The package is currently under development and can be installed from the repository. The package requires Python 3.12. The package can be installed using the following commands. This will install the package with the CPU dependencies --- i.e., the cpu version of [PyTorch](https://pytorch.org/) and [scikit-learn](https://scikit-learn.org/stable/index.html) for clustering.

Create a virtual environment, e.g. using conda.

```bash
conda create -n torchgeodemo python=3.12
conda activate torchgeodemo
```

Install the package via `pip`.

```bash
pip install torchgeodemo
```

Alternatively, the package can be installed with the GPU dependencies if available and [RAPIDS](https://docs.rapids.ai/) for the clustering.

```bash
pip install --extra-index-url=https://pypi.nvidia.com torchgeodemo[gpu]
```


## Usage

The commands below illustrate how to use the Command-Line Interface (CLI) to run the tool using the example configuration file `example/example.yml` and data (eight random blobs in sixteen dimensions). The `-t` flag is used to train the autoencoder and `l` to create the latent representation. The `-s` flag is used to search for the best k. The `-c` flag is used to run clustering using k-means. If available, the cuml backend is used for clustering. The `-v` flag is optional and is used to display the progress of the process. 

Train the autoencoder.

```bash
torchgeodemo -tv example/example.yml
```

Create latent representation using on the previously trained autoencoder. Note that this will load a model from disk, and thus it will rais a warning message, as that can result in **arbitrary code execution**. Do it only if you got the file from a **trusted** source -- e.g., a model file you trained yourself, using the command above.

```bash
torchgeodemo -lv example/example.yml
```

Alternatively, you can train the autoencoder and create the latent representation in one go. In this case, the autoencoder will still be saved, but the latent representation will be created directly with the model in memory (rather than loading from the disk).

```bash
torchgeodemo -tlv example/example.yml
```

Run clustering in test mode to search for best k. Add `-r` for RAPIDS backend, default is scikit-learn.

```bash
torchgeodemo -sv example/example.yml
```

Run clustering using k-means. Add `-r` for RAPIDS backend, default is scikit-learn.

```bash
torchgeodemo -cv example/example.yml
```

Alternatively, you can run everything in one go as well.

```bash
torchgeodemo -tlscv example/example.yml
```

For a more concrete example, you can test the tool using the 2021 OAC data available from [Jakub Wyszomierski's repo](https://github.com/jakubwyszomierski/OAC2021-2). Download the [Clean data](https://liveuclac-my.sharepoint.com/:f:/g/personal/zcfajwy_ucl_ac_uk/Eqd1EV2WgOFJmZ7kLx-oDYMBdxqNe9IJmli6M8S-e91F0g?e=M9wh5j) used to create the [2021 OAC](https://data.cdrc.ac.uk/dataset/output-area-classification-2021), unzip the file and set the value of `data: source` to the location of one of the file datasets on your computer. It is advisable to normalise the data before training the autoencoder, e.g., using min-max scaling. Please increase the number of epochs and the number of clustering iteration to get meaninful results.



## Unit Tests

If you want to run the unit tests, you can install the package in editable mode.

```bash
# Clone the repository
gh repo clone sdesabbata/torchgeodemo
cd torchgeodemo

# Install the package
pip install -e .
```

Then run the tests.

```bash
python -m pytest tests/ -v
```


## Acknowledgement

Many thanks to [Owen Goodwin](https://github.com/ogoodwin505), [Pengyuan Liu](https://github.com/PengyuanLiu1993) and [Alex Singleton](https://github.com/alexsingleton) for their collaboration on this project and for testing the pre-alpha versions of the tool.
