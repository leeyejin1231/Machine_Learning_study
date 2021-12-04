import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def numerical_solution(self, x, y, epochs, batch_size, lr, optim, batch_gradient=False):

        """
        The numerical solution of Linear Regression

        [Inputs]
            x : input for linear regression. Numpy array of (N, D)
            y : label of data x. Numpy array of (N, )
            epochs : epochs.
            batch_size : size of the batch.
            lr : learning rate.
            optim : optimizer.

        [Output]
            None

        """

        self.W = self.W.reshape(-1)
        num_data = len(x)
        num_batch = int(np.ceil(num_data / batch_size))

        for epoch in range(epochs):
            if batch_gradient:
                # batch gradient descent
                
                loss_vector = None
                grad = None
                s = 0

                for i in range(num_data):
                    s += (y[i] - x[i].dot(self.W))*x[i]

                grad = -s/num_data

                self.W = optim.update(self.W, grad, lr)
            else:
                # mini-batch stochastic gradient descent
                for batch_index in range(num_batch):
                    batch_x = x[batch_index*batch_size:(batch_index+1)*batch_size]
                    batch_y = y[batch_index*batch_size:(batch_index+1)*batch_size]

                    num_samples_in_batch = len(batch_x)

                    loss_vector = None
                    grad = None
                    s = 0;

                    for j in range(num_samples_in_batch):
                        s += (batch_y[j] - batch_x[j].dot(self.W))*batch_x[j]

                    grad = -s/num_samples_in_batch

                    self.W = optim.update(self.W, grad, lr)

    def analytic_solution(self, x, y):
        """
        The analytic solution of Linear Regression

        [Inputs]
            x : input for linear regression. Numpy array of (N, D)
            y : label of data x. Numpy array of (N, )

        [Output]
            None
        """

        X_transpose = np.transpose(x)
        self.W = np.linalg.inv(X_transpose.dot(x))      #(X^T * X)^-1
        self.W = self.W.dot(X_transpose)                 #(X^T * X)^-1 * X^T
        self.W = self.W.dot(y)                           #(X^T * X)^-1 * X^T * y


    def eval(self, x):
        pred = None

        """
        Evaluation Function
        [Input]
            x : input for linear regression. Numpy array of (N, D)

        [Outputs]
            pred : prediction for 'x'. Numpy array of (N, )
        """

        pred = np.dot(x, self.W)
        
        return pred
