# Usage

The commands below illustrate how to use the Command-Line Interface (CLI) to run the tool using the example configuration file `examples/example.yml` and data (eight random blobs in sixteen dimensions). The `-t` flag is used to train the autoencoder and `l` to create the latent representation. The `-s` flag is used to search for the best k. The `-c` flag is used to run clustering using k-means. If available, the cuml backend is used for clustering. The `-v` flag is optional and is used to display the progress of the process. 

Train the autoencoder.

```bash
deepgeodemo -tv examples/example.yml
```

Create latent representation using on the previously trained autoencoder. Note that this will load a model from disk, and thus it will rais a warning message, as that can result in **arbitrary code execution**. Do it only if you got the file from a **trusted** source -- e.g., a model file you trained yourself, using the command above.

```bash
deepgeodemo -lv examples/example.yml
```

Alternatively, you can train the autoencoder and create the latent representation in one go. In this case, the autoencoder will still be saved, but the latent representation will be created directly with the model in memory (rather than loading from the disk).

```bash
deepgeodemo -tlv examples/example.yml
```

Run clustering in test mode to search for best k. Add `-r` for RAPIDS backend, default is scikit-learn.

```bash
deepgeodemo -sv examples/example.yml
```

Run clustering using k-means. Add `-r` for RAPIDS backend, default is scikit-learn.

```bash
deepgeodemo -cv examples/example.yml
```

Alternatively, you can run everything in one go as well.

```bash
deepgeodemo -tlscv examples/example.yml
```

For a more concrete example, you can test the tool using the 2021 OAC data available from [Jakub Wyszomierski's repo](https://github.com/jakubwyszomierski/OAC2021-2). Download the [Clean data](https://liveuclac-my.sharepoint.com/:f:/g/personal/zcfajwy_ucl_ac_uk/Eqd1EV2WgOFJmZ7kLx-oDYMBdxqNe9IJmli6M8S-e91F0g?e=M9wh5j) used to create the [2021 OAC](https://data.cdrc.ac.uk/dataset/output-area-classification-2021), unzip the file and set the value of `data: source` to the location of one of the file datasets on your computer. It is advisable to normalise the data before training the autoencoder, e.g., using min-max scaling. Please increase the number of epochs and the number of clustering iteration to get meaninful results.
