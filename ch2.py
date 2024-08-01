import numpy as np
import os
import scipy
import sklearn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
# 1)
class Percepron (object):
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.erors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int (update != 0.0)
            self.erors_.append(errors)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return np.where(self.net_input(X)>= 0.0, 1, -1)
    
v1 = np.array([1,2,3])
v2 = 0.5 * v1
np.arccos(v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
#2
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('URL:', s)
df = pd.read_csv(s,header=None,encoding='utf-8')
print(df.tail())
#3
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100,[0,2]].values
plt.scatter(X[:50,0], X[:50,1], color='r',marker='o',label='Setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='b',marker='x',label='Versicolor')
plt.xlabel('Długość działki [cm]')
plt.ylabel('Długość płatka [cm]')
plt.legend(loc='upper left')
plt.show()
#4
ppn = Percepron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1,len(ppn.erors_)+1),ppn.erors_,marker='o')
plt.xlabel('Epoki')
plt.ylabel('Liczba aktualizacji')
plt.show()
#5
def plot_decision_regions(X,y,classfier, resolution=0.22):
    markers = ('s','x', 'o', '^', 'V')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:,0].min() - 1, X[:, 0].max()+1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    Z = classfier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2, Z, alpha = 0.3, cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1],alpha=0.8,c=colors[idx], marker=markers[idx],label=cl,edgecolors='black')
#6
plot_decision_regions(X,y,classfier=ppn)
plt.xlabel('Długość działki [cm]')
plt.ylabel('Długość płatka [cm]')
plt.legend(loc='upper left')
plt.show()


### Adaline
class AdalineGD(object):
    """
    Parametry
    -----------
    eta: zmiennoprzcinkowy
        Współczynnik uczenia (w zakresie pomiędzy 0.0 i 1.0)
    n_iter: liczba całkowita
        Liczba przebiegów po zestawie uczącym
    random_state: liczba całkowita
        Ziarno generatora liczb losowych służące do inicjowania lsowych wag
    
    Atrybuty
    -----------
    w_: jednowymiarowa tablica
        Wagi po dopasowaniu
    cost_: lista
        Suma kwadratów błędów (wartość funkcji kosztu) w każdej epoce.
    """
    def __init__(self, eta = 0.01, n_iter = 50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """Trenowanie za pomocą danych uczących
        Parametry
        ------------
        X: {tablicopodobny}, wymiary =[n_przykładów,n_cech]
            Wektory uczenia,
            gdzie n_przykładów oznacza liczbe przykładów, a
            n_cech liczbę cech
        y: tablicopodobny, wymiary = [n_przykładów]
            Wartości docelowe
        Zwraca
        ------------
        self: obiekt
        
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta*X.T.dot(errors)
            self.w_ [0] += self.eta*errors.sum()
            cost = (errors**2).sum() /2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """Oblicza całkowite pobudzenie"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def activation (self, X):
        """Oblicza liniową funkcję aktywacji"""
        return X
    def predict(self, X):
        """Zwraca etykietę klas po wykonaniu skoku jednostkowego"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    
fig, ax = plt.subplots(nrows=1, ncols = 2, figsize=(10,4))
ada1 = AdalineGD(n_iter=10,eta=0.01).fit(X,y)
ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker='o')
ax[0].set_xlabel('Epoki')
ax[0].set_ylabel('Log (suma kwadratów błędów)')
ax[0].set_title('Adaline - Współczynnik uczenia 0,01')
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X,y)
ax[1].plot(range(1,len(ada2.cost_)+1),ada2.cost_,marker = 'o')
ax[1].set_xlabel('Epoki')
ax[1].set_ylabel('Suma kwadratów błędów')
ax[1].set_title('Adaline - Współczynnik uczenia 0,0001')
plt.show()

#standardization
X_std = np.copy(X)
X_std [:,0] = (X_std[:,0]-X_std[:,0].mean())/X_std[:,0].std()
X_std [:,1] = (X_std[:,1] - X_std[:,1].mean())/X_std[:,1].std()


ada_gd = AdalineGD(n_iter=15, eta = 0.01)
ada_gd.fit(X_std,y)
plot_decision_regions(X_std,y,classfier=ada_gd)
plt.title('Adaline - Gradient prosty')
plt.xlabel('Długość działki [standaryzowana]')
plt.ylabel('Długość płatka [standaryzowana]')
plt.legend(loc ='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1,len(ada_gd.cost_)+1), ada_gd.cost_, marker='o')
plt.xlabel('Epoki')
plt.ylabel('Suma kwadratów błędów')
plt.tight_layout()
plt.show()