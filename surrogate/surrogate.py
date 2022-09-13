import numpy as np
from sklearn.cluster import KMeans

class Gaussian(object):
    @staticmethod
    def func(x, c, s):
        x = np.atleast_2d(x)
        c = np.atleast_2d(c)
        return np.exp(-1 / (2 * s ** 2) * np.sum((x - c) ** 2,axis=1))

    @staticmethod
    def dfunc(x, c, s):
        x = np.atleast_2d(x)
        c = np.atleast_2d(c)
        ret = []
        for i in range(len(x)):
            dx = []
            for j in range(len(x[i])):
                dx.append(Gaussian.func(x[i], c, s).squeeze() * -1 / (s ** 2) * (x[i,j] - c[0,j]))
            ret.append(np.array(dx))
        return np.array(ret)

    @staticmethod
    def hfunc(x, c, s):
        x = np.atleast_2d(x)
        c = np.atleast_2d(c)
        rets = []
        for i in range(len(x)):
            ret = np.zeros((len(x[i]),len(x[i])))
            for j in range(len(x[i])):
                for k in range(len(x[i])):
                    ret[j,k] = Gaussian.func(x[i], c, s).squeeze() * 1 / (s ** 4) * (x[i,j] - c[0,j]) * (x[i,k] - c[0,k])
                    if j == k:
                        ret[j,k] -= Gaussian.func(x[i], c, s).squeeze() / ( s ** 2 )
            rets.append(ret)
        return np.array(rets)

class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, k=2, kernel=Gaussian):
        self.k = k
        self.kernel = kernel
        self.w = np.random.randn(k)

    def _activations(self, X, func):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.k))
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = func(x,c,self.std)
        return G

    def fit(self, X, y):
        cluster = KMeans(n_clusters=self.k)
        cluster.fit(X)

        self.centers = cluster.cluster_centers_

        dMax = np.max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
        #self.std = dMax / np.sqrt(2*self.k)
        self.std = 0.02

        H = np.zeros((X.shape[0], self.k))
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                H[xi, ci] = self.kernel.func(x, c, self.std)

        self.w = np.dot(np.linalg.pinv(H), y)

    def predict(self, X):
        ys_pred = []
        H = np.zeros((X.shape[0], self.k))
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                H[xi, ci] = self.kernel.func(x, c, self.std)
        for i in range(X.shape[0]):
            F = H[i].dot(self.w)
            ys_pred.append(F)
        return np.vstack(ys_pred)

    def predictive_gradients(self, X):
        dys_pred = []
        for i in range(X.shape[0]):
            dh = []
            for c in self.centers:
                dh.append(self.kernel.dfunc(X[i],c,self.std))
            dh = np.vstack(dh).T
            F = dh.dot(self.w)
            dys_pred.append(F)
        return np.vstack(dys_pred)

    def predictive_hessian(self,X):
        hys_pred = []
        for i in range(X.shape[0]):
            hh = []
            for c in self.centers:
                hh.append(self.kernel.hfunc(X[i],c,self.std))
            hh = np.vstack(hh).T
            F = hh.dot(self.w)
            hys_pred.append(F)
        return np.array(hys_pred)
