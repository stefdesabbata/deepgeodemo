# Installation

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


## Unit Tests

If you want to run the unit tests, you can install the package in editable mode.

```bash
# Clone the repository
gh repo clone stefdesabbata/deepgeodemo
cd deepgeodemo

# Install the package
pip install -e .
```

Then run the tests.

```bash
python -m pytest tests/ -v
```
