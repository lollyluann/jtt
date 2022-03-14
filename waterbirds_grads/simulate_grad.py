import torch, math
from tqdm import tqdm
import numpy as np
from utils import simul_x_y_a, add_outliers, plot_sample, plot_decision, plot_grad, plot_3d
from metrics import group_metrics
from sklearn.metrics import accuracy_score, confusion_matrix

device = torch.device("cuda")

def full_detach(x):
    return x.squeeze().detach().cpu().numpy()
'''
mu_mult = 2.
cov_mult = 1.
skew = 5.

train_prop_mtx = [[0.4, 0.1],[0.4, 0.1]]
train_x, train_a, train_y = simul_x_y_a(train_prop_mtx, n=1000, mu_mult=mu_mult, 
                                        cov_mult=cov_mult, skew=skew, outliers=True)
plot_sample(train_x, train_a, train_y, title='Train')

test_prop_mtx = [[0.25, 0.25],[0.25, 0.25]]
test_x, test_a, test_y = simul_x_y_a(test_prop_mtx, n=1000, mu_mult=mu_mult, 
                                     cov_mult=cov_mult, skew=skew, outliers=False)
plot_sample(test_x, test_a, test_y, title='Test')

train_x, train_y, text_x, test_y = map(torch.tensor, (train_x, train_y, test_x, test_y))
train_x.requires_grad_()
test_x = torch.tensor(test_x)
test_x.requires_grad_()
train_y.double()

test_biased_x, test_biased_a, test_biased_y = simul_x_y_a(train_prop_mtx, n=1000, 
                                                          mu_mult=mu_mult, cov_mult=cov_mult,
                                                          skew=skew)

'''

train_x = torch.tensor(np.load("../train_data_resnet50.npy"), dtype=torch.float64, device=device, requires_grad=True)
train_y = torch.tensor(np.load("../train_data_y_resnet50.npy"), device=device)
train_l = np.load("../train_data_l_resnet50.npy")

a = train_x.shape[1]
weights = torch.randn(a, 1, device=device, dtype=torch.double)/math.sqrt(a)
weights.requires_grad_()
bias = torch.zeros(1, requires_grad=True, device=device, dtype=torch.double)

def model(xb):
    sig = torch.nn.Sigmoid().to(device)
    return sig(xb @ weights + bias)

def export_grads(x, y, model, optimizer, data_name, loss_func=torch.nn.BCELoss()):
    weight_traingrad, bias_traingrad = [], []
    for i, pt in enumerate(x):
        ptpred = torch.squeeze(model(pt[None, :]), dim=1)
        loss2 = loss_func(ptpred, y[i:i+1].double())
        loss2.backward()
        weight_grad = full_detach(weights.grad)
        bias_grad = full_detach(bias.grad)
        weight_traingrad.append(weight_grad.copy())
        bias_traingrad.append(bias_grad.copy())
        with torch.no_grad():
            weights.grad.zero_()
            bias.grad.zero_()
    
    weight_traingrad = np.array(weight_traingrad)
    bias_traingrad = np.array(bias_traingrad)

    grads = np.append(weight_traingrad, bias_traingrad[np.newaxis].T, axis=1)
    print(weight_traingrad.shape, bias_traingrad.shape, grads.shape)
    save_dir = "weight_bias_grads_"+data_name+".npy"
    np.save(save_dir, grads)

loss_func = torch.nn.BCELoss()
optimizer = torch.optim.Adam([weights, bias], lr=0.001)
#lr = 0.05
epochs = 100

weight_traingrad, input_traingrad, bias_traingrad = [], [], []
weight_testgrad, input_testgrad, bias_testgrad = [], [], []

for epoch in tqdm(range(epochs)):
    pred = model(train_x).squeeze()
    loss = loss_func(pred, train_y.double())
    optimizer.zero_grad()
    loss.backward()
    input_traingrad = full_detach(train_x.grad)
    optimizer.step()
   
    '''with torch.no_grad():
        weights -= weights.grad * lr
        bias -= bias.grad * lr
        weights.grad.zero_()
        bias.grad.zero_()
    '''

    if epoch==epochs-1:
        '''weight_traingrad, bias_traingrad = [], []
        for i, pt in enumerate(train_x):
            ptpred = torch.squeeze(model(pt[None, :]), dim=1)
            loss2 = loss_func(ptpred, train_y[i:i+1].double())
            loss2.backward()
            weight_grad = full_detach(weights.grad)
            bias_grad = full_detach(bias.grad)
            weight_traingrad.append(weight_grad.copy())
            bias_traingrad.append(bias_grad.copy())
            with torch.no_grad():
                weights.grad.zero_()
                bias.grad.zero_()'''
        export_grads(train_x, train_y, model, optimizer, "train")
    
    '''
    testloss = loss_func(model(test_x), test_y)
    testloss.backward()
    input_testgrad = test_x.grad.squeeze().detach().cpu().numpy()
    weight_testgrad = weights.grad.squeeze().detach().cpu().numpy()
    bias_testgrad = bias.grad.squeeze().detach().cpu().numpy()
    weights.grad.zero_()
    bias.grad.zero_()'''


def predict(model, data):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    return full_detach(model(data).round())

def evaluate(model, data, labels, groups):
    predictions = predict(model, data)
    acc = accuracy_score(labels, predictions)

    for ig in [0, 1, 2, 3, 4]:
        indices = [i for i in range(len(labels)) if (groups[i]==ig)]
        print("Group", ig, ":", len(indices), "samples")
        print(accuracy_score(labels[indices], predictions[indices]))
    print("Overall accuracy", acc)
    print(confusion_matrix(labels, predictions))

## Base classifier
#base_predict = predict(model, train_x)
#print('Baseline')
#_ = group_metrics(full_detach(train_y), base_predict, test_a, label_protected=1, label_good=0)
#print('Test biased accuracy', np.mean(predict(model, test_biased_x) == test_biased_y))

## Base ideal classifier
'''base_lr_ideal = LogisticRegression(solver='liblinear', fit_intercept=True)
base_lr_ideal.fit(test_x, test_y)
base_predict_ideal = base_lr_ideal.predict(test_x)
print('\nBaseline IDEAL')
_ = group_metrics(test_y, base_predict_ideal, test_a, label_protected=1, label_good=0)
'''
#plot_decision(full_detach(test_x), test_a, full_detach(test_y), lambda x: predict(model, x), title='Log Reg')
#plot_decision(test_x, test_a, test_y, lambda x: base_lr_ideal.predict_proba(x)[:,1], title='Log Reg IDEAL')

print("Training data accuracies")
evaluate(model, train_x, full_detach(train_y), train_l)

#weight_traingrad = np.array(weight_traingrad)
#bias_traingrad = np.array(bias_traingrad)
#plot_grad(full_detach(train_x), train_a, full_detach(train_y), input_traingrad, title="TrainGradInput")
#plot_grad(full_detach(train_x), train_a, full_detach(train_y), weight_traingrad, title="TrainGradWeight")
#plot_3d(full_detach(train_x), full_detach(train_y), train_a, weight_traingrad, bias_traingrad, title="TrainGradWeightsBias")
#plot_grad(full_detach(train_x), train_a, full_detach(train_y), bias_traingrad, title="TrainGradBias")
#plot_grad(full_detach(test_x), test_a, full_detach(test_y), input_testgrad, title="TestGrad")

#grads = np.append(weight_traingrad, bias_traingrad[np.newaxis].T, axis=1)
#print(weight_traingrad.shape, bias_traingrad.shape, grads.shape)
#save_dir = "weight_bias_grads.npy"
#np.save(save_dir, grads)

test_x = torch.tensor(np.load("../test_data_resnet50.npy"), dtype=torch.float64, device=device, requires_grad=True)
test_y = torch.tensor(np.load("../test_data_y_resnet50.npy"), device=device)
test_l = np.load("../test_data_l_resnet50.npy")

print("Test data accuracies")
evaluate(model, test_x, full_detach(test_y), test_l)

val_x = torch.tensor(np.load("../val_data_resnet50.npy"), dtype=torch.float64, device=device, requires_grad=True)
val_y = torch.tensor(np.load("../val_data_y_resnet50.npy"), device=device)
val_l = np.load("../val_data_l_resnet50.npy")

print("Validation data accuracies")
evaluate(model, val_x, full_detach(val_y), val_l)

export_grads(val_x, val_y, model, optimizer, "val")
export_grads(test_x, test_y, model, optimizer, "test")



