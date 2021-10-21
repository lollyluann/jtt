import torch, math
import numpy as np
from utils import simul_x_y_a, plot_sample, plot_decision, plot_grad
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

train_x, train_y, text_x, test_y = map(torch.tensor, (train_x, train_y, test_x, test_y))
train_x.requires_grad_()
test_x = torch.tensor(test_x)
test_x.requires_grad_()

test_biased_x, test_biased_a, test_biased_y = simul_x_y_a(train_prop_mtx, n=1000, 
                                                          mu_mult=mu_mult, cov_mult=cov_mult,
                                                          skew=skew)

a = train_x.shape[1]
weights = torch.randn(a, 2, dtype=torch.double)/math.sqrt(a)
weights.requires_grad_()
bias = torch.zeros(2, requires_grad=True, dtype=torch.double)

def model(xb):
    return torch.nn.LogSoftmax(dim=1)(xb @ weights + bias)

loss_func = torch.nn.NLLLoss()
lr = 0.05
epochs = 30

weight_traingrad, input_traingrad, bias_traingrad = [], [], []
weight_testgrad, input_testgrad, bias_testgrad = [], [], []

for epoch in range(epochs):
    pred = model(train_x)
    loss = loss_func(pred, train_y)
    loss.backward()
    input_traingrad = train_x.grad.squeeze().detach().cpu().numpy()
    weight_traingrad = weights.grad.squeeze().detach().cpu().numpy()
    bias_traingrad = bias.grad.squeeze().detach().cpu().numpy()
    
    with torch.no_grad():
        weights -= weights.grad * lr
        bias -= bias.grad * lr
        weights.grad.zero_()
        bias.grad.zero_()
    
    testloss = loss_func(model(test_x), test_y)
    testloss.backward()
    input_testgrad = test_x.grad.squeeze().detach().cpu().numpy()
    weight_testgrad = weights.grad.squeeze().detach().cpu().numpy()
    bias_testgrad = bias.grad.squeeze().detach().cpu().numpy()
    weights.grad.zero_()
    bias.grad.zero_()

def full_detach(x):
    return x.squeeze().detach().cpu().numpy()

def predict(model, data):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    return full_detach(model(data).argmax(dim=1))

## Base classifier
base_predict = predict(model, test_x)
print('Baseline')
_ = group_metrics(full_detach(test_y), base_predict, test_a, label_protected=1, label_good=0)
print('Test biased accuracy', np.mean(predict(model, test_biased_x) == test_biased_y))

## Base ideal classifier
'''base_lr_ideal = LogisticRegression(solver='liblinear', fit_intercept=True)
base_lr_ideal.fit(test_x, test_y)
base_predict_ideal = base_lr_ideal.predict(test_x)
print('\nBaseline IDEAL')
_ = group_metrics(test_y, base_predict_ideal, test_a, label_protected=1, label_good=0)
'''
plot_decision(full_detach(test_x), test_a, full_detach(test_y), lambda x: full_detach(model(torch.tensor(x)))[:,1], title='Log Reg')
#plot_decision(test_x, test_a, test_y, lambda x: base_lr_ideal.predict_proba(x)[:,1], title='Log Reg IDEAL')

print(weight_traingrad.shape, input_traingrad.shape, bias_traingrad)
plot_grad(full_detach(train_x), train_a, full_detach(train_y), input_traingrad, title="TrainGrad")
plot_grad(full_detach(test_x), test_a, full_detach(test_y), input_testgrad, title="TestGrad")
