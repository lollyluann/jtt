from sklearn.linear_model import LogisticRegression
import numpy as np

train_x = np.load("../train_data_resnet50.npy")
train_y = np.load("../train_data_y_resnet50.npy")

val_x = np.load("../val_data_resnet50.npy")
val_y = np.load("../val_data_y_resnet50.npy")

lr = LogisticRegression()
lr2 = lr.fit(train_x, train_y)
lr2.predict(val_x)
print(lr2.score(val_x, val_y))
