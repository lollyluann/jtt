import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

compute_dists = True
do_dbscan = False
do_agg = False
bias_separate = True
dim_red = "PCA"
overwrite = False

data_dir = "weight_bias_grads.npy"
grads = np.load(data_dir)
train_l = np.load("../train_data_l_resnet50.npy")
print("Loaded gradients of shape", grads.shape)
print(grads.shape[0], "data points,", grads.shape[1], "gradients")

if compute_dists:
    distance_matrix = pdist(grads, metric="euclidean")
    avg_distance = distance_matrix.mean()
    print("Avg pairwise distance:", avg_distance)
    _ = plt.hist(distance_matrix, bins='auto')
    plt.title("Histogram of pairwise distances")
    plt.savefig("histogram_dists.pdf")

    def square_to_condensed(i, j, n):
        if i==j: return 0
        if i<j:
            i, j = j, i
        return n*j - j*(j+1)//2 + i - 1 - j

    # compute average/histogram within and cross group distances
    sums_groups = np.zeros((5,5))
    counts_groups = np.zeros((5,5))
    n = grads.shape[0]
    for a in tqdm(range(grads.shape[0])):
        for b in range(a+1, grads.shape[0]):
            sums_groups[train_l[a], train_l[b]] += distance_matrix[square_to_condensed(a, b, n)]
            sums_groups[train_l[b], train_l[a]] += distance_matrix[square_to_condensed(a, b, n)]
            counts_groups[train_l[a], train_l[b]] += 1
            counts_groups[train_l[b], train_l[a]] += 1
    avgs_groups = sums_groups/counts_groups
    print("Avg group distances")
    print(avgs_groups)
    fig = plt.figure()
    ax = sns.heatmap(avgs_groups)
    plt.savefig("group_distances_heatmap.pdf")

if dim_red == "MDS":
    print("Dimensionality reduction via MDS")
    if not os.path.exists("mds_data.npy"):
        plotdata = MDS(n_components=3).fit_transform(grads)
        np.save("mds_data.npy", plotdata)
    else:
        plotdata = np.load("mds_data.npy")
elif dim_red == "PCA":
    print("Dimensionality reduction via PCA")
    if not os.path.exists("pca_data.npy") or overwrite:
        if bias_separate:
            plotdata = PCA(n_components=2).fit_transform(grads)
            plotdata = np.append(plotdata, np.array([grads[:,-1]]).T, axis=1)
        else:
            plotdata = PCA(n_components=3).fit_transform(grads)
        np.save("pca_data.npy", plotdata)
    else:
        plotdata = np.load("pca_data.npy")

if do_dbscan:
    eps_options = [avg_distance*i/100 for i in range(10, 90, 10)]
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

if do_agg:
    agg = cluster.AgglomerativeClustering(n_clusters=None, distance_threshold=175)
    #agg = cluster.AgglomerativeClustering(n_clusters=4)
    labels = agg.fit_predict(grads)
    print(Counter(labels))
    print(agg.n_clusters_)
    print(confusion_matrix(train_l, labels))

# read in true labels
'''train_y = np.load("../train_data_y_resnet50.npy")
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
'''

fig = plt.figure()
ax = Axes3D(fig)
scattered = ax.scatter(plotdata[:,0], plotdata[:,1], plotdata[:,2], c=train_l, cmap="Spectral")
ax.text2D(0.05, 0.95, dim_red+": 4 groups + outliers", transform=ax.transAxes)
legend = ax.legend(*scattered.legend_elements(), loc="upper right", title="Groups")
ax.add_artist(legend)
plt.savefig(dim_red+"_ground_truth_memberships.pdf")
