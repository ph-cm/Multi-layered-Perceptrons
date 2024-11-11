import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.datasets import make_classification
import numpy as np
#you can generate anothers seeds
np.random.seed(0)
import random
import matplotlib.cm as cm


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

#Network Class
class Net:
    def __init__(self):
        self.layers = []
    
    def add(self, l):
        self.layers.append(l)
    
    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x
    
    def backward(self, z):
        for l in self.layers[::-1]:
            z = l.backward(z)
        return z
    
    def update(self, lr):
        for l in self.layers:
            if 'update' in l.__dir__():
                l.update(lr)

net = Net()
net.add(Linear(2,2))
net.add(Softmax())
loss = CrossEntropyLoss()

def get_loss_acc(x,y,loss=CrossEntropyLoss()):
    p = net.forward(x)
    l = loss.forward(p,y)
    pred = np.argmax(p,axis=1)
    acc = (pred==y).mean()
    return l,acc

print("Initial loss={}, accuracy={} ".format(*get_loss_acc(train_x,train_labels)))

def train_epoch(net, train_x, train_labels, loss=CrossEntropyLoss(), batch_size=4, lr=0.1):
    for i in range(0,len(train_x), batch_size):
        xb = train_x[i:i+batch_size]
        yb = train_labels[i:i+batch_size]
        
        p = net.forward(xb)
        l = loss.forward(p, yb)
        dp = loss.backward(l)
        dx = net.backward(dp)
        net.update(lr)

train_epoch(net,train_x,train_labels)

print("Final loss={}, accuracy={}: ".format(*get_loss_acc(train_x, train_labels)))
print("Test loss ={}, accuracy={}: ".format(*get_loss_acc(test_x, test_labels)))

#Plotting the Training Process
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_decision_boundary(net, train_x, train_labels, fig, ax):
    ax.clear()
    
    # Gera a grade de contorno
    x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 1
    y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    # Prepara os pontos da grade para a predição
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = net.forward(grid_points)
    
    # Calcula a probabilidade para a classe 1 (caso binário)
    if predictions.shape[1] == 2:
        Z = predictions[:, 1].reshape(xx.shape)
    else:
        Z = predictions.argmax(axis=1).reshape(xx.shape)
    
    # Desenha o contorno de probabilidade
    levels = np.linspace(0, 1, 40)
    cs = ax.contourf(xx, yy, Z, alpha=0.7, levels=levels, cmap='viridis')
    fig.colorbar(cs, ax=ax, ticks=[0, 0.5, 1])

    # Mapa de cores para os pontos de treino
    c_map = [cm.coolwarm(x) for x in np.linspace(0.0, 1.0, len(set(train_labels)))]
    colors = [c_map[label] for label in train_labels]
    ax.scatter(train_x[:, 0], train_x[:, 1], c=colors, edgecolors='k', s=50, alpha=0.8)

    fig.canvas.draw()
    plt.pause(0.01)

def train_and_plot(n_epoch, net, loss=CrossEntropyLoss(), batch_size=4, lr=0.01):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].set_xlim(0, n_epoch + 1)
    ax[0].set_ylim(0, 1)

    train_acc = np.empty((n_epoch, 3))
    train_acc[:] = np.NAN
    valid_acc = np.empty((n_epoch, 3))
    valid_acc[:] = np.NAN

    plt.ion()

    for epoch in range(1, n_epoch + 1):
        # Treina a rede para a época atual
        train_epoch(net, train_x, train_labels, loss, batch_size, lr)
        
        # Calcula a perda e acurácia para o conjunto de treinamento
        tloss, taccuracy = get_loss_acc(train_x, train_labels, loss)
        train_acc[epoch - 1, :] = [epoch, tloss, taccuracy]
        
        # Calcula a perda e acurácia para o conjunto de validação
        vloss, vaccuracy = get_loss_acc(test_x, test_labels, loss)
        valid_acc[epoch - 1, :] = [epoch, vloss, vaccuracy]

        ax[0].set_ylim(0, max(max(train_acc[:, 2]), max(valid_acc[:, 2])) * 1.1)

        # Atualiza o gráfico de progresso do treinamento
        plot_training_progress(train_acc[:, 0], (train_acc[:, 2], valid_acc[:, 2]), fig, ax[0])
        
        # Atualiza o gráfico da fronteira de decisão
        # Durante o treinamento
        plot_decision_boundary(net, train_x, train_labels, fig, ax[1], draw_colorbar=True)

        
        fig.canvas.draw()
        plt.pause(0.01)

    plt.ioff()
    plt.show()
    return train_acc, valid_acc

def plot_training_progress(x, y_data, fig, ax):
    ax.clear()
    styles = ['k--', 'g-']
    for i, y in enumerate(y_data):
        ax.plot(x, y, styles[i])
    ax.legend(['training accuracy', 'validation accuracy'], loc='upper left')

def plot_decision_boundary(net, train_x, train_labels, fig, ax, draw_colorbar=False):
    ax.clear()  # Limpa apenas o contorno sem sobrepor os pontos

    # Gera a grade de contorno
    x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 1
    y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    
    # Prepara os pontos da grade para a predição
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = net.forward(grid_points)

    # Calcula a probabilidade para a classe 1 (caso binário)
    if predictions.shape[1] == 2:
        Z = predictions[:, 1].reshape(xx.shape)
    else:
        Z = predictions.argmax(axis=1).reshape(xx.shape)
    
    # Desenha o contorno de probabilidade apenas uma vez
    levels = np.linspace(0, 1, 40)
    cs = ax.contourf(xx, yy, Z, levels=levels, cmap='viridis', alpha=0.6)
    
    # Atualiza ou cria a barra de cor
    if draw_colorbar and not hasattr(ax, 'colorbar'):
        ax.colorbar = fig.colorbar(cs, ax=ax, ticks=[0, 0.5, 1])
    
    # Desenha os pontos de treino sem limpar a cada iteração
    c_map = [cm.coolwarm(x) for x in np.linspace(0.0, 1.0, len(set(train_labels)))]
    colors = [c_map[label] for label in train_labels]
    ax.scatter(train_x[:, 0], train_x[:, 1], c=colors, edgecolors='k', s=50, alpha=0.8)

    fig.canvas.draw()
    plt.pause(0.01)


# Função de treinamento e plotagem
net = Net()
net.add(Linear(2,2))
net.add(Softmax())

res = train_and_plot(30,net,lr=0.005)

#Multi-Layered Models
class Tanh:
    def forward(self,x):
        y = np.tanh(x)
        self.y = y
        return y
    def backward(self,dy):
        return(1.0-self.y**2)*dy
net = Net()
net.add(Linear(2,10))
net.add(Tanh())
net.add(Linear(10,2))
net.add(Softmax())
loss = CrossEntropyLoss()

res = train_and_plot(30, net, lr=0.01)
