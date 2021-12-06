import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

data_dir = "weight_bias_grads.npy"
grads = np.load(data_dir)
print("Loaded gradients of shape", grads.shape)
print(grads.shape[0], "data points,", grads.shape[1], "gradients")

distance_matrix = pdist(grads, metric="euclidean")
avg_distance = distance_matrix.mean()
print("Avg pairwise distance:", avg_distance)
_ = plt.hist(distance_matrix, bins='auto')
plt.title("Histogram of pairwise distances")
plt.savefig("histogram_dists.pdf")

eps_options = [avg_distance*i/100 for i in range(10, 90, 10)]
plotdata = MDS(n_components=3).fit_transform(grads)

for ep in eps_options:
    dbscan = cluster.DBSCAN(eps=ep, min_samples=5)
    clustered = dbscan.fit_predict(grads)
    num_clusters = np.unique(clustered).size-1
    print("eps={} yielded {} clusters".format(ep, num_clusters))
    print(Counter(clustered))

    fig = plt.figure()
    ax = Axes3D(fig)
    scattered = ax.scatter(plotdata[:,0], plotdata[:,1], plotdata[:,2], c=clustered, cmap="Spectral")
    ax.text2D(0.05, 0.95, str(num_clusters) + " clusters + outliers", transform=ax.transAxes)
    plt.savefig("cluster_ep_" + str(ep) + ".pdf")

agg = cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=175)
#agg = cluster.AgglomerativeClustering(n_clusters=4)
labels = agg.fit_predict(grads)
print(Counter(labels))
print(agg.n_clusters_)


# read in true labels
train_y = np.load("../train_data_y_resnet50.npy")
train_g = np.load("../train_data_g_resnet50.npy")
print(train_g)
# y: 0 is land, 1 is water
# g: 0 is land, 1 is water

label = []
for i in range(len(train_y)):
    if train_y[i]==0:
        if train_g[i]==0:
            label.append(0)
        else: label.append(1)
    else:
        if train_g[i]==0:
            label.append(2)
        else: label.append(3)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(label, labels))
