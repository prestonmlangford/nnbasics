import nnmodel as nn
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model




def plot_decision_boundary(model, data):
    X, y = data
    
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model.forward((np.c_[xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=np.ravel(y), cmap=plt.cm.Spectral)
    
def visualize_dataset(data):
    X,Y = data
    plt.scatter(X[0, :], X[1, :], c=np.ravel(Y), s=40, cmap=plt.cm.Spectral)
    plt.draw()

def make_2d_dataset(size,kind):
    N = size
    
    if kind is "blobs":
        X,Y = sklearn.datasets.make_blobs(
            n_samples=N, 
            #random_state=9, 
            n_features=2, 
            centers=5,
            cluster_std=1)
        Y = Y%2
    elif kind is "circles":        
        X,Y = sklearn.datasets.make_circles(
            n_samples=N, 
            factor=.5, 
            noise=.1)
    elif kind is "moons":
        X,Y = sklearn.datasets.make_moons(
            n_samples=N, 
            noise=.2)
    elif kind is "gaussian":    
        X,Y = sklearn.datasets.make_gaussian_quantiles(
            mean=None, 
            cov=0.5, 
            n_samples=N, 
            n_features=2, 
            n_classes=2, 
            shuffle=True, 
            random_state=None)
            
    # sklearn has a different convention for dataset shapes 
    X = X.T
    Y = Y.reshape(1, Y.shape[0])
    
    #normalize data
    X = X-np.mean(X,axis=1,keepdims=True)
    X = X/np.std(X,axis=1,keepdims=True)
    
    return X,Y 


data = make_2d_dataset(400,"circles")
visualize_dataset(data)

mod = nn.NNModel(
    [
        nn.AnnealingFullyConnected(2,32),
        nn.AnnealingFullyConnected(32,32),
        nn.AnnealingFullyConnected(32,16),
        nn.AnnealingFullyConnected(16,1),
    ]
)

mod.train(
    data = data,
    iterations = 1000,
    alpha = 0.01,
    lambd = 0.9,
    beta_v = 0.9,
    beta_s = 0.9,
)

plot_decision_boundary(mod,data)

# keeps windows open after program finishes
plt.show()