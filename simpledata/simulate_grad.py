import torch
import numpy as np
from utils import simul_x_y_a, plot_sample, plot_decision
from sklearn.linear_model import LogisticRegression
from metrics import group_metrics

mu_mult = 2.
cov_mult = 1.
skew = 5.

train_prop_mtx = [[0.4, 0.1],[0.4, 0.1]]
train_x, train_a, train_y = simul_x_y_a(train_prop_mtx, n=1000, mu_mult=mu_mult, 
                                        cov_mult=cov_mult, skew=skew)
plot_sample(train_x, train_a, train_y, title='Train')

test_prop_mtx = [[0.25, 0.25],[0.25, 0.25]]
test_x, test_a, test_y = simul_x_y_a(test_prop_mtx, n=1000, mu_mult=mu_mult, 
                                     cov_mult=cov_mult, skew=skew)
plot_sample(test_x, test_a, test_y, title='Test')

train_x, train_a, train_y, text_x, test_a, test_y = map(torch.tensor, (train_x, train_a, train_y, test_x, test_a, test_y))
train_x.requires_grad_()
test_x.requires_grad_()

test_biased_x, test_biased_a, test_biased_y = simul_x_y_a(train_prop_mtx, n=1000, 
                                                          mu_mult=mu_mult, cov_mult=cov_mult,
                                                          skew=skew)

a = train_x.shape[1]
weights = torch.randn(a, 2)/(a**0.5)
weights.requires_grad_()
bias = torch.zeros(2, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll
lr = 0.1
epochs = 10

for epoch in range(epochs):
    for i in range(train_x.shape[0]):
        pred = model(train_x)
        loss = loss_func(train_x, train_y)
        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

## Base classifier
base_lr = LogisticRegression(solver='liblinear', fit_intercept=True)
base_lr.fit(train_x, train_y)
base_predict = base_lr.predict(test_x)
print('Baseline')
_ = group_metrics(test_y, base_predict, test_a, label_protected=1, label_good=0)
print('Test biased accuracy', np.mean(base_lr.predict(test_biased_x) == test_biased_y))

## Base ideal classifier
base_lr_ideal = LogisticRegression(solver='liblinear', fit_intercept=True)
base_lr_ideal.fit(test_x, test_y)
base_predict_ideal = base_lr_ideal.predict(test_x)
print('\nBaseline IDEAL')
_ = group_metrics(test_y, base_predict_ideal, test_a, label_protected=1, label_good=0)

plot_decision(test_x, test_a, test_y, lambda x: base_lr.predict_proba(x)[:,1], title='Log Reg')
plot_decision(test_x, test_a, test_y, lambda x: base_lr_ideal.predict_proba(x)[:,1], title='Log Reg IDEAL')
