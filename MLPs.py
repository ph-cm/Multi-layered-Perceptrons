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
    ylabels = ['$\mathcal{L}_{abs}}$ (absolute loss)', '$\mathcal{L}_{sq}$ (squared loss)'],
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
    ylabels =  ['$\mathcal{L}_{0-1}}$ (0-1 loss)', '$\mathcal{L}_{log}$ (logistic loss)'],
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

