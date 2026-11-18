import numpy as np


class LinearRegression:
    def __init__(self,lr=0.01):
        self.weight = np.zeros((1, 1))
        self.bias = 0.0
        self.lr=lr
        self.loss_history = []
        self.epoch_record = []
    def predict(self,x):
        y_pred=np.dot(x,self.weight)+self.bias
        return y_pred
    def train(self,x,y,epochs):
        n=x.shape[0]
        for epoch in range(epochs):
            y_pred = self.predict(x)
            deta = y_pred - y
            pingfang = deta ** 2
            loss = np.sum(pingfang) /n

            dw = (2 / n) * x.T @ deta
            db = 2*(np.sum(deta)) / n
            self.weight = self.weight - self.lr * dw
            self.bias = self.bias - self.lr * db
            if epoch%100==0:
               self.loss_history.append(loss)
               self.epoch_record.append(epoch)
               print("epoch:", epoch, "loss:", loss)
               print("weight:", self.weight, "bias:", self.bias)