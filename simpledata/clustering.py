import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

data_dir = "weight_bias_grads.npy"
grads = np.load(data_dir)
print("Loaded gradients of shape", grads.shape)

eps_options = [0.05, 0.1, 0.25, 0.5, 1]

for ep in tqdm(eps_options):
    dbscan = cluster.DBSCAN(eps=ep, min_samples=5)
    clustered = dbscan.fit_predict(grads)
    fig = plt.figure()
    ax = Axes3D(fig)
    scattered = ax.scatter(grads[:,0], grads[:,1], grads[:,2], c=clustered, cmap="Spectral")
    legend = ax.legend(scattered.legend_elements(), loc="lower left", title="Cluster")
    ax.add_artist(legend)
    plt.savefig("cluster_ep_" + str(ep) + ".pdf")
