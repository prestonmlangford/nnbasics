# Package imports
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, load_planar_dataset, load_extra_datasets



def cross_entropy(Y,Yh):
    
    # n is the number of training examples
    n = Y.shape[1]
    
    # to deal with numerical issues, limit Yh in (0,1)
    epsilon = 1e-10
    Yh = np.maximum(Yh,epsilon)
    Yh = np.minimum(Yh,1-epsilon)
    
    cost = -(1/n)*(np.dot(Y,np.log(Yh.T))+np.dot(1-Y,np.log(1-Yh.T)))
    cost = np.squeeze(cost)
    #dYh = ((1-Yh)*Y-Yh*(1-Y))/(Yh*(1-Yh))#-(Y/Yh-(1-Y)/(1-Yh))
    #dYh = (Yh-Y)/(Yh*(1-Yh))
    dYh = -(Y/Yh-(1-Y)/(1-Yh))
    
    # returns cost and derivative of cost WRT Yh
    return cost, dYh

class FullyConnected:
    def __init__(self,inputs,outputs,activation):
        self.activation = activation
        
        # "He Initialization"
        W_scale = np.sqrt(1/inputs)
        self.W = np.random.randn(outputs,inputs)*W_scale
        self.B = np.zeros((outputs,1))
        
        # these will get set later.  
        # Setting them here to satisfy python class constructor
        # d is the gradient term,
        # v is the momentum term, 
        # s is the second moment for RMS prop
        self.dW = np.zeros(self.W.shape)
        self.vW = np.zeros(self.W.shape)
        self.sW = np.zeros(self.W.shape)
        
        self.dB = np.zeros(self.B.shape)
        self.vB = np.zeros(self.B.shape)
        self.sB = np.zeros(self.B.shape)
        
        # n is the number of training examples for the current iteration
        # values of X and Y will be over written during the forward pass
        # Y is the output of the layer
        # X is the input to the layer
        # these values get cached for the backwards pass
        self.n = 1
        self.dH  = np.zeros((outputs,1))
        self.X  = np.zeros((inputs,1))
    
    def forward(self,X,update_cache = True):
        
        # n is the number of training examples
        # Z[outputs,n] = W[outputs,inputs] x X[inputs,n] + B[outputs,1]
        # B will broadcast over the second dimension
        Z = np.dot(self.W,X)+self.B
        
        # choose activation set by initialization
        if self.activation == "tanh":    
            Y = np.tanh(Z)
            dH = 1-np.square(Y)
        elif self.activation == "relu":
            Y = np.maximum(0,Z)
            dH = np.double(Y>0)
        elif self.activation == "sigmoid":
            Y = 1/(1+np.exp(-Z))
            dH = Y*(1-Y)
        elif self.activation == "softsign":
            gamma = 0.01
            den = gamma+abs(Z)
            Y = Z/den
            dH = gamma/(den*den)
        else:
            raise ValueError("FullyConnected layer activation type not set")
        
        if update_cache:
            # cache these for the backward pass to calculate gradients
            self.X = X
            self.dH = dH
        
        return Y
    
    
    def backward(self,dY):
        
        # n is the number of training examples
        n = dY.shape[1]
        
        # dZ[outputs,n] = dY[outputs,n] * dG[outputs,n]
        dZ = dY*self.dH
        
        # dW[outputs,inputs] = dZ[outputs,n] x dX'[n,inputs]
        dW = np.dot(dZ,self.X.T)/n
        
        # dB[outputs,1] = average over n dZ[outputs,n]
        dB = np.sum(dZ,axis=1,keepdims=True)/n
        
        # dX[inputs,n] = W'[inputs,outputs] x dZ[outputs,n]
        dX = np.dot(self.W.T,dZ)
        
        # save these for update
        self.dW = dW
        self.dB = dB
        self.n  = n
        
        return dX
        
    
    def update(self,alpha,lambd,beta_v,beta_s,iterations):
        
        self.vW = beta_v*self.vW+(1-beta_v)*self.dW
        self.vB = beta_v*self.vB+(1-beta_v)*self.dB
        
        # the step below is different than Adam.  I use STD instead of RMS
        self.sW = beta_s*self.sW+(1-beta_s)*np.square(self.dW-self.vW)
        self.sB = beta_s*self.sB+(1-beta_s)*np.square(self.dB-self.vB)
        
        t = iterations+1
        vWc = self.vW/(1-np.power(beta_v,t))
        vBc = self.vB/(1-np.power(beta_v,t))
        sWc = self.sW/(1-np.power(beta_s,t))
        sBc = self.sB/(1-np.power(beta_s,t))
        
        epsilon = 1e-8
        Wstep = alpha*vWc/(epsilon+np.sqrt(sWc))
        Bstep = alpha*vBc/(epsilon+np.sqrt(sBc))
        
        # uses "weight decay".  This is equivalent to L2 regularization
        # cost does not include the regularization cost, because why bother?
        decay = 1-alpha*lambd/self.n
        self.W = self.W*decay - Wstep
        self.B = self.B*decay - Bstep

class AnnealingFullyConnected:

    def softsign(self,x):
        d = self.gamma + np.abs(x)
        return x/d

    def dsoftsign(self,x):
        d = self.gamma + np.abs(x)
        d2 = d*d
        return self.gamma/d2

    def __init__(self,inputs,outputs):
        self.gamma = 1

        # He Initialization
        # incorporate bias into the weights.  Have one bias unit for every weight
        self.w = np.random.randn(outputs,2*inputs)*np.sqrt(1/inputs)
        self.W = self.softsign(np.zeros(self.w.shape))

        # these will get set later.  
        # Setting them here to satisfy python class constructor
        # d is the gradient term,
        # v is the momentum term, 
        # s is the second moment for RMS prop
        self.dw = np.zeros(self.w.shape)
        self.vw = np.zeros(self.w.shape)
        self.sw = np.zeros(self.w.shape)
        
        # n is the number of training examples for the current iteration
        # values of X and Y will be over written during the forward pass
        # Y is the output of the layer
        # X is the input to the layer
        # these values get cached for the backwards pass
        self.n = 1
        self.dhY  = np.zeros((outputs,1))
        self.XB  = np.zeros((2*inputs,1))
    
    def forward(self,X,update_cache = True):
        
        # n is the number of training examples
        # Z[outputs,n] = W[outputs,inputs] x X[inputs,n] + B[outputs,1]
        # B will broadcast over the second dimension
        XB = np.append(X,np.ones(X.shape),axis=0)
        W = self.softsign(self.w)
        Z = np.dot(W,XB)
        Y = self.softsign(Z)
        
        if update_cache:
            # cache these for the backward pass to calculate gradients
            self.XB = XB
            self.dhY = self.dsoftsign(Z)
            self.W = W

        return Y
    
    
    def backward(self,dY):
        
        # n is the number of training examples
        n = dY.shape[1]
        
        # dZ[outputs,n] = dY[outputs,n] * dH[outputs,n]
        dZ = dY*self.dhY
        
        # dW[outputs,inputs] = dZ[outputs,n] x dX'[n,inputs]
        dW = np.dot(dZ,self.XB.T)/n
        dhw = self.dsoftsign(self.w)
        
        # dX[inputs,n] = W'[inputs,outputs] x dZ[outputs,n]
        dX = np.dot(self.W.T,dZ)
        
        # save these for update
        self.dw = dW*dhw
        self.n  = n
        
        return dX[0:dX.shape[0]//2,:]
        
    
    def update(self,alpha,lambd,beta_v,beta_s,iterations):
        
        self.vw = beta_v*self.vw+(1-beta_v)*self.dw
        # the step below is different than Adam.  I use STD instead of RMS
        self.sw = beta_s*self.sw+(1-beta_s)*np.square(self.dw-self.vw)

        t = iterations+1
        vwc = self.vw/(1-np.power(beta_v,t))
        swc = self.sw/(1-np.power(beta_s,t))
        
        epsilon = 1e-8
        wstep = alpha*vwc/(epsilon+np.sqrt(swc))
        
        # uses "weight decay".  This is equivalent to L2 regularization
        # cost does not include the regularization cost, because why bother?
        decay = 1-alpha*lambd/self.n
        self.w = self.w*decay - wstep

        anneal = 0.01
        # update gamma 
        if iterations > 700:
            self.gamma = self.gamma*(1-anneal)
        if (iterations % 100) == 0:
            print(self.gamma)


# PMLFIXME implement dropout
class NNModel:
    def __init__(self,layers):
        self.layers = layers


    def forward(self,X,update_cache = True):
        for layer in self.layers: 
            X = layer.forward(X,update_cache)
        return (X+1)/2#PMLFIXME just for annealed layer

    def backward(self,dY):
        for layer in reversed(self.layers): 
            dY = layer.backward(dY)
    
    def update(self,alpha,lambd,beta_v,beta_s,iterations):
        for layer in self.layers: 
            layer.update(alpha,lambd,beta_v,beta_s,iterations)
    
    def train(self,data,iterations,alpha,lambd,beta_v,beta_s):
        for i in range(0,iterations):
            X,Y = data
            
            Yh = self.forward(X)
            
            cost, dYh = cross_entropy(Y,Yh)
            
            self.backward(dYh)
            
            self.update(alpha,lambd,beta_v,beta_s,i)
            
            if (i%100) == 0:
                print("Cost after " + str(i) + " iterations: " + str(cost))
    

