import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.neural_network import MLPClassifier

class HyperParamOpt:
    def __init__(self,dataset='mnist_784'):
        self.n_var = 7
        self.n_obj = 2
        self.minibatch = 50
        self.train_samples = 50
        self.X, self.y = fetch_openml(dataset, version=1, return_X_y=True, as_frame=False)
        random_state = check_random_state(0)
        permutation = random_state.permutation(self.X.shape[0])
        self.X = self.X[permutation]
        self.y = self.y[permutation]
        self.X = self.X.reshape((self.X.shape[0], -1))
        _, self.X_test, _, self.y_test = train_test_split(
            self.X, self.y, train_size=self.train_samples, test_size=100)
        self.X_train = self.X[0:self.minibatch,:]
        self.y_train = self.y[0:self.minibatch]
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)

        self.X_test = self.scaler.transform(self.X_test)
        self.var_bound = np.array([[1e-9,1.0],[1e-9,1.0],[0,1.0 - 1e-9],[0,1.0 - 1e-9],[0,4 - 1e-9],[0,10 - 1e-9],[1, 8 - 1e-9]])

    def evaluate(self, xs):
        activations = {0:'identity',1:'logistic',2:'tanh',3:'relu'}
        xs = np.atleast_2d(xs)
        ret = np.zeros((xs.shape[0],2))
        for i in range(xs.shape[0]):
            x = xs[i,:]
            clf = MLPClassifier(solver='adam',learning_rate='constant', random_state=1,
                    alpha = x[0],learning_rate_init=x[1],beta_1=x[2],beta_2=x[3],activation=activations[int(x[4])],
                                hidden_layer_sizes=(int(x[5]+1), int(np.power(2,int(x[6])))),
            )
            #print(f'hyper param: alpha = {x[0]},learning_rate_init={x[1]},beta_1={x[2]},beta_2={x[3]},'
            #      f'activation={activations[int(x[4])]},hidden_layer_sizes=({int(x[5]+1)}, {int(np.power(2,int(x[6])))})')
            clf.fit(self.X_train,self.y_train)
            ret[i,0] = 1-clf.score(self.X_test,self.y_test)
            sum = 0.0
            for coef in clf.coefs_:
                sum += np.abs(np.sum((coef)))
            ret[i,1] = sum
        return ret