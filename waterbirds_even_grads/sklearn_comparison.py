from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

model = "pretrained-18"
remove_out = False

train_x = np.load("../train_data_resnet_" + model + ".npy")
train_x = np.load("../resnet-18_1layertrain.npy")
train_y = np.load("../train_data_y_resnet_" + model + ".npy")
train_g = np.load("../train_data_g_resnet_" + model + ".npy")
train_l = np.load("../train_data_l_resnet_" + model + ".npy")

val_x = np.load("../val_data_resnet_" + model + ".npy")
val_x = np.load("../resnet-18_1layerval.npy")
val_y = np.load("../val_data_y_resnet_" + model + ".npy")
val_g = np.load("../val_data_g_resnet_" + model + ".npy")
val_l = np.load("../val_data_l_resnet_" + model + ".npy")

test_x = np.load("../test_data_resnet_" + model + ".npy")
test_x = np.load("../resnet-18_1layertest.npy")
test_y = np.load("../test_data_y_resnet_" + model + ".npy")
test_g = np.load("../test_data_g_resnet_" + model + ".npy")
test_l = np.load("../test_data_l_resnet_" + model + ".npy")

print("train data shape", train_x.shape)

def get_group_accs(pred, groups):
    n_groups = np.unique(groups).size
    print(n_groups, "groups")
    for i in range(n_groups):
        correct = i//2
        acc = accuracy_score(pred[groups==i], np.full(pred[groups==i].size, correct))
        print("Group", i, "accuracy:", acc)

def remove_outliers(x, y, g, l):
    return x[l!=4], y[l!=4], g[l!=4], l[l!=4]

if remove_out:
    print("Filtering out outliers")
    train_x, train_y, train_g, train_l = remove_outliers(train_x, train_y, train_g, train_l)
    val_x, val_y, val_g, val_l = remove_outliers(val_x, val_y, val_g, val_l)
    print("train data size", train_x.shape)

print("true y", train_y[0:10], train_y[70:80])

lr = LogisticRegression()
lr2 = lr.fit(train_x, train_y)

val_out = lr2.predict(val_x)
print("Validation avg acc:", lr2.score(val_x, val_y))
get_group_accs(val_out, val_g)

test_out = lr2.predict(test_x)
print("Test avg acc:", lr2.score(test_x, test_y))
get_group_accs(test_out, test_g)

