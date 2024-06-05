# pandas и numpy
import pandas as pd
import numpy as np
# визуализация
import matplotlib.pyplot as plt
import seaborn as sns
from lib.utils import t_quant
#считаем коэффициенты линейной регрессии
class multi_regression:
    def __init__(self, X, Y):
        try:
            assert(len(self.X) == len(self.Y)) # проверка на равенство длины массивов, потом нужно будет сделать обработчик исключений
        except:
            print("Количество наблюдений в X и Y не совпадает")
            print("Невозможно продолжить выполнение программы")
            exit(1)
        self.X = X
        self.Y = Y
        self.rgc = self.regression_coefs()
        self.df = len(self.Y) - len(self.X[0]) - 1
        self.Y_pr = np.dot(self.X, self.rgc)
        self.Y_mn = self.Y.mean()
        self.ESS = sum([(y_i - Y_mn)**2 for y_i in self.Y_pr])
        self.RSS = sum([(Y[i] - Y_pr[i])**2 for i in range(len(Y))])
    def regression_coefs(self):
        xtx = np.dot(np.transpose(self.X),self.X)
        xty = np.dot(np.transpose(self.X),self.Y)
        coefs = np.linalg.solve(xtx, xty)
        self.rgc = coefs
        return
    def T_test(self, j):
        X_T = np.transpose(X)
        r_model = multi_regression(X_T[:j] + X_T[j+1:], Y)
        F = ((r_model.RSS - self.RSS)/(r_model.df - self.df))/(self.RSS/self.df)
        
X = np.array([[1,2,-1,3], [-1,1,0,0], [0,2,1,0], [0,0,1,-1],[1,-1,2,1]])
Y = np.transpose(np.array([1,2,3,4,5]))

