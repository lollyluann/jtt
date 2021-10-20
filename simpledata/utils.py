import numpy as np
import matplotlib.pyplot as plt

def simul_x_y_a(prop_mtx, n=100, mu_mult=1., cov_mult=0.5, skew=2., rotate=0):
    
    mu_y0_a0 = np.array([1.,1.])*mu_mult
    mu_y0_a1 = np.array([5., 7.])*mu_mult
    mu_y1_a0 = np.array([1.,3.])*mu_mult
    mu_y1_a1 = np.array([3., 7.])*mu_mult
    
    # mu_y0_a0 = np.array([1.,1.])*mu_mult
    # mu_y0_a1 = np.array([5., 7.])*mu_mult
    # mu_y1_a0 = np.array([-1,1.])*mu_mult
    # mu_y1_a1 = np.array([3., 7.])*mu_mult
    
    
    mu = [[mu_y0_a0, mu_y0_a1], [mu_y1_a0, mu_y1_a1]]
    
    cov_y0_a0 = np.array([skew,1.])*cov_mult
    cov_y0_a1 = np.array([1.,skew])*cov_mult
    cov_y1_a0 = np.array([skew,1.])*cov_mult
    cov_y1_a1 = np.array([1.,skew])*cov_mult
    
    # cov_y0_a0 = np.array([1.,skew])*cov_mult
    # cov_y0_a1 = np.array([1.,skew])*cov_mult
    # cov_y1_a0 = np.array([1.,skew])*cov_mult
    # cov_y1_a1 = np.array([1.,skew])*cov_mult
    
    cov = [[cov_y0_a0, cov_y0_a1], [cov_y1_a0, cov_y1_a1]]
    
    data_x = []
    data_y = []
    data_a = []
    
    for y in [0,1]:
        for a in [0,1]:
            n_ya = int(n*prop_mtx[y][a])
            data_y += n_ya*[y]
            data_a += n_ya*[a]
            data_x.append(np.random.normal(loc=mu[y][a], scale=np.sqrt(cov[y][a]), size=(n_ya,2)))
            
            if a == 1 and rotate > 0:
                mean = data_x[-1].mean(axis=0)
                data_x[-1] = (data_x[-1]-mean) @ rotation(rotate) + mean
    
    order = np.random.permutation(len(data_y))
    
    data_x = np.vstack(data_x)[order]
    data_x = np.sqrt(data_x - data_x.min(axis=0))
    # if rotate > 0:
    #     mean = data_x.mean(axis=0)
    #     data_x = (data_x-mean) @ rotation(rotate) + mean
        
    data_y = np.array(data_y)[order]
    data_a = np.array(data_a)[order]
    
    return data_x, data_a, data_y

def rotation(angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def plot_sample(data_x, data_a, data_y, ax=None, title=None, show=True):
    markers = ['o' , 'x']
    colors = ['r','b']
    
    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
    for y in [0,1]:
        for a in [0,1]:
            x_ya = data_x[np.logical_and(data_a==a, data_y==y)]
            ax.scatter(x_ya[:,0],x_ya[:,1], c=colors[y], marker=markers[a], s=75, label='y=%d, a=%d' % (y,a))
    plt.legend(loc='upper left', fontsize=15)
    plt.grid()
    if title is not None:
        plt.title(title, fontsize=15)
        
    if show:
        plt.show()
    plt.savefig("plot_sample.pdf")
    
    return ax

def plot_decision(data_x, data_a, data_y, decision_f, title=None):
    
    # Decision colormap
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    # ax = plot_sample(data_x, data_a, data_y, ax=None, title=title, show=False)
    h = .02
    # cm = plt.cm.RdBu
    cm = plt.cm.Spectral
    x_min, x_max = data_x[:, 0].min() - .5, data_x[:, 0].max() + .5
    y_min, y_max = data_x[:, 1].min() - .5, data_x[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = decision_f(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    plot_sample(data_x, data_a, data_y, ax=ax, title=title, show=False)
    plt.xlim([x_min, x_max])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.savefig("plot_decision.pdf")
