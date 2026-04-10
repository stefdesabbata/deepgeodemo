import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

latent_df = pd.read_csv('examples/example_geodemo__to4emb_dcc_v0_1__latent.csv')
clusters_df = pd.read_csv('examples/example_geodemo__to4emb_dcc_v0_1__kmeans_clusters.csv')
clusters_df['test_0_1_k008_now'] = clusters_df['test_0_1_k008_now'].astype(str)

results_df = pd.merge(latent_df, clusters_df, on=['CL', 'ID'])  

sns.pairplot(
    results_df.drop(columns=['ID', 'test_0_1_k008_now']),
    hue='CL',
    plot_kws={'s': 1}
    )
plt.savefig('examples/example_geodemo__to4emb_dcc_v0_1__latent.png')

sns.pairplot(
    results_df.drop(columns=['ID', 'CL']),
    hue='test_0_1_k008_now',
    plot_kws={'s': 1}
    )
plt.savefig('examples/example_geodemo__to4emb_dcc_v0_1__kmeans_clusters.png')
