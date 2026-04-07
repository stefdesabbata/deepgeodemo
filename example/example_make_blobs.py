import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

random_state = 20210321

n_samples = 10000
n_features = 16
n_clusters = 8

sample_data, sample_labels = make_blobs(
    n_samples=n_samples,
    centers=n_clusters,
    n_features=n_features,
    cluster_std=0.05,
    center_box=(0.0, 1.0),
    random_state=random_state
    )

print('Data shape:', sample_data.shape)
print('Save plot and csv file...')

sample_labels_df = pd.DataFrame({
    'CL': [f'CL{int(i)+1:02d}' for i in sample_labels],
    'ID': [f'X{int(i)+1:05d}' for i in range(n_samples)]
    })
sample_data_df = pd.DataFrame(
    sample_data,
    columns=[f'feature_{i}' for i in range(n_features)]
    )
sample_df = pd.concat([sample_labels_df, sample_data_df], axis=1)

print(sample_df.head())
sample_df.to_csv('example/example.csv', index=False)

sns.pairplot(
    sample_df.drop(columns=['ID']),
    hue='CL',
    plot_kws={'s': 1}
    )
plt.savefig('example/example_input.png')
