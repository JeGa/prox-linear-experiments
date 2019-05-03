import logging
import numpy as np
import matplotlib.pyplot as plt

import modelbased.data.spirals


class LinearSVM:
    def __init__(self):
        self.dataset = modelbased.data.spirals.load('datasets/binary-spirals')

    def f(self, u, x):
        return x.T.dot(u)

    def P(self, u, x, y, lam):
        A = self.f(u, x)

        return x.shape[0] * self.svm_ova(A, y).sum(0) + lam * 0.5 * (u ** 2).sum()

    def D1(self, a, x, y, lam):
        n = x.shape[1]
        Z = x.dot(y * a)

        return -lam * 0.5 * (((1 / (lam * n)) * Z) ** 2).sum()

    def gD1(self, a, x, y, lam):
        n = x.shape[1]

        return -(1 / (lam * n ** 2)) * x.dot(y).T.dot(x).dot(y * a)

    def proj(self, x):
        x[x >= 1] = 1
        x[x <= 0] = 0

        return x

    def svm_ova(self, a, y):
        return np.maximum(0, 1 - y * a)

    def run(self):
        x, y = self.dataset[:len(self.dataset)]

        x = x.T
        y[y[:, 1] == 1, 1] = -1
        y = np.expand_dims(y.sum(1), 1)

        x = x[:, 0:50]
        y = y[0:50]

        plt.figure(0)

        pos = (y == 1).squeeze()
        neg = (y == -1).squeeze()
        plt.title('gt')
        plt.scatter(x[0, pos], x[1, pos])
        plt.scatter(x[0, neg], x[1, neg])
        plt.show()





        u0 = np.ones((x.shape[0], 1))
        a0 = np.ones((x.shape[1], 1))





        yp = np.sign(self.f(u0, x))

        pos = yp == 1
        neg = yp == -1

        plt.figure(1)
        plt.title('init')
        plt.scatter(x[0, pos.squeeze()], x[1, pos.squeeze()])
        plt.scatter(x[0, neg.squeeze()], x[1, neg.squeeze()])
        plt.show()







        print(self.P(u0, x, y, 1))
        print(self.D1(a0, x, y, 1))
        self.gD1(a0, x, y, 1)

        a = a0
        tau = 0.002
        lam = 1

        for i in range(200):
            a = a + tau * self.gD1(a, x, y, lam)
            a = self.proj(a)

            print(self.D1(a, x, y, lam))

        n = x.shape[1]
        w = (1 / (n * lam)) * x.dot(a * y)

        yp = np.sign(self.f(w, x))

        pos = yp == 1
        neg = yp == -1

        plt.figure(2)
        plt.title('pred')
        plt.scatter(x[0, pos.squeeze()], x[1, pos.squeeze()])
        plt.scatter(x[0, neg.squeeze()], x[1, neg.squeeze()])

        plt.show()

        pass


def run():
    cls = LinearSVM()

    cls.run()
