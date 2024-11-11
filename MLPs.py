import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.datasets import make_classification
import numpy as np
#you can generate anothers seeds
np.random.seed(0)
import random

#making sample dataset with two parameters
n = 100
X, Y = make_classification(n_samples = n, n_features = 2, n_redundant=0, n_informative=2, flip_y=0.2)

X = X.astype(np.float32)
Y = Y.astype(np.int32)

#split into train and test dataset
train_x, test_x = np.split(X, [n*8//10])
train_labels, test_labels = np.split(Y, [n*8//10])

def plot_dataset(suptitle, features, labels):
    #preparing the plot
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(suptitle, fontsize = 16)
    ax.set_xlabel('$x_i[0]$ -- (feature 1)')
    ax.set_ylabel('$x_i[1]$ -- (feature 2)')
    
    colors = ['r' if l else 'b' for l in labels]
    ax.scatter(features[:, 0], features[:, 1], marker='o', c = colors, s = 100, alpha = 0.5)
    fig.show()
    
plot_dataset('Scatterplot of the training data', train_x, train_labels)
plt.show()
    
print(train_x[:5])
print(train_labels[:5])

#loss function for regression
#helper function for plotting various loss functions

def plot_loss_functions(suptitle, functions, ylabels, xlabel):
    fig, ax = plt.subplots(1, len(functions), figsize=(9, 3))
    plt.subplots_adjust(bottom=0.2, wspace=0.4)
    fig.suptitle(suptitle)
    for i, fun in enumerate(functions):
        ax[i].set_xlabel(xlabel)
        if len(ylabels) > i:
            ax[i].set_ylabel(ylabels[i])
        ax[i].plot(x, fun)
    plt.show()

x = np.linspace(-2, 2, 101)
plot_loss_functions(
    suptitle = 'Common loss functions for regresion',
    functions = [np.abs(x), np.power(x, 2)],
    ylabels = [r'$\mathcal{L}_{abs}$ (absolute loss)', r'$\mathcal{L}_{sq}$ (squared loss)'],
    xlabel = '$y - f(x_i)$'
)

#loss function for classification
x = np.linspace(0, 1, 100)
def zero_one(d):
    if d < 0.5:
        return 0
    return 1
zero_one_v = np.vectorize(zero_one)

def logistic_loss(fx):
    #assume y == 1
    return -np.log(fx)

plot_loss_functions(
    suptitle = 'Common loss functions for classification (class = 1)',
    functions = [zero_one_v(x), logistic_loss(x)],
    ylabels = [r'$\mathcal{L}_{0-1}$ (0-1 loss)', r'$\mathcal{L}_{log}$ (logistic loss)'],
    xlabel = '$p$'
)

#Neural Network Architecture
class Linear:
    def __init__(self, nin, nout):
        self.W = np.random.normal(0, 1.0/np.sqrt(nin), (nout, nin))
        self.b = np.zeros((1, nout))
    def forward(self, x):
        return np.dot(x, self.W.T) + self.b
    
net = Linear(2, 2)
net.forward(train_x[0:5])
    
#Softmax turning outputs into probabilities
class Softmax:
    def forward(self,z):
        zmax = z.max(axis=1, keepdims=True)
        expz = np.exp(z-zmax)
        Z = expz.sum(axis=1, keepdims=True)
        return expz / Z
softmax = Softmax()
softmax.forward(net.forward(train_x[0:10]))

#Cross Entropy Loss
def plot_cross_ent():
    p = np.linspace(0.01, 0.99, 101) #estimated probability p(y/x)
    cross_ent_v = np.vectorize(cross_ent)
    f3, ax = plt.subplots(1, 1, figsize = (8, 3))
    l1, = plt.plot(p, cross_ent_v(p, 1), 'r--')
    l2, = plt.plot(p, cross_ent_v(p, 0), 'r-')
    plt.legend([l1, l2], ['$y = 1$', '$y = 0$'], loc = 'upper center', ncol = 2)
    plt.xlabel(r'$\hat{p}(y|x)$', size = 18)
    plt.ylabel(r'$\mathcal{L}_{CE}$', size = 18)
    plt.show()
    
def cross_ent(prediction, ground_truth):
    t = 1 if ground_truth > 0.5 else 0
    return -t * np.log(prediction) - (1 - t) * np.log(1 - prediction)
plot_cross_ent()

class CrossEntropyLoss:
    def forward(self,p,y):
        self.p = p
        self.y = Y
        p_of_y = p[np.arange(len(y)), y]
        log_prob = np.log(p_of_y)
        return -log_prob.mean()
cross_ent_loss = CrossEntropyLoss()
p = softmax.forward(net.forward(train_x[0:10]))
cross_ent_loss.forward(p, train_labels[0:10])

z = net.forward(train_x[0:10])
p = softmax.forward(z)
loss = cross_ent_loss.forward(p, train_labels[0:10])
print(loss)

#Loss Minimization Problem nd Network Training
class Linear:
    def __init__(self, nin, nout):
        self.W = np.random.normal(0, 1.0/np.sqrt(nin), (nout, nin))
        self.b = np.zeros((1,nout))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
    
    def forward(self, x):
        self.x = x 
        return np.dot(x, self.W.T) + self.b
    
    def backward(self, dz):
        dx = np.dot(dz, self.W)
        dW = np.dot(dz.T, self.x)
        db = dz.sum(axis = 0)
        self.dW = dW
        self.db = db
        return dx

    def update(self, lr):
        self.W -= lr*self.dW
        self.b -= lr*self.db

class Softmax:
    def forward(self,z):
        self.z = z
        zmax = z.max(axis = 1, keepdims = True)
        expz = np.exp(z - zmax)
        Z = expz.sum(axis = 1, keepdims = True)
        return expz / Z
    def backward(self, dp):
        p = self.forward(self.z)
        pdp =p * dp
        return pdp - p * pdp.sum(axis = 1, keepdims = True)
    
class CrossEntropyLoss:
    def forward(self,p,y):
        self.p = p
        self.y = y
        print("Shape of self.p:", self.p.shape)  # Debugging: Check shape of probabilities
        print("Values in self.y:", self.y)
        p_of_y = p[np.arange(len(y)), y]
        log_prob = np.log(p_of_y)
        return -log_prob.mean()
    def backward(self,loss):
        dlog_softmax = np.zeros_like(self.p)
        dlog_softmax[np.arange(len(self.y)), self.y] -= 1.0/len(self.y)
        return dlog_softmax / self.p

#Training the Model
lin = Linear(2,2)
softmax = Softmax()
cross_ent_loss = CrossEntropyLoss()

learning_rate = 0.1

pred = np.argmax(lin.forward(train_x),axis=1)
acc = (pred==train_labels).mean()
print("Initial accuracy: ",acc)

batch_size=4
for i in range(0,len(train_x),batch_size):
    xb = train_x[i:i+batch_size]
    yb = train_labels[i:i+batch_size]
    
    # forward pass
    z = lin.forward(xb)
    p = softmax.forward(z)
    loss = cross_ent_loss.forward(p,yb)
    
    # backward pass
    dp = cross_ent_loss.backward(loss)
    dz = softmax.backward(dp)
    dx = lin.backward(dz)
    lin.update(learning_rate)
    
pred = np.argmax(lin.forward(train_x),axis=1)
acc = (pred==train_labels).mean()
print("Final accuracy: ",acc)