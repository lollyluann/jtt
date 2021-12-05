from sklearn.linear_model import LogisticRegression
import numpy as np

train_x = np.load("../train_data_resnet50.npy")
train_y = np.load("../train_data_y_resnet50.npy")

lr = LogisticRegression()
lr2 = lr.fit(train_x, train_y)
lr2.predict(train_x)
print(lr2.score(train_x, train_y))
