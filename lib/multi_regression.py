# pandas и numpy
import pandas as pd
import numpy as np
# визуализация
import matplotlib.pyplot as plt
import seaborn as sns
# функция рассчета критических значений
from lib.utils import f_critical_value as fcv
# считаем коэффициенты линейной регрессии
class multi_regression:
    def __init__(self, X, Y):
        try:
            assert(len(X) == len(Y)) # проверка на равенство длины массивов, потом нужно будет сделать обработчик исключений
        except:
            print("Количество наблюдений в X и Y не совпадает")
            print("Невозможно продолжить выполнение программы")
            exit(1)
        self.X = X
        self.Y = Y
        self.regression_coefs()                                       # коэффиценты регрессии
        self.df = len(self.Y) - len(self.X[0]) - 1                               # количество степеней свободы
        self.Y_pr = np.dot(self.X, self.rgc)
        self.Y_mn = self.Y.mean()
        self.TSS = sum([(y_i - self.Y_mn)**2 for y_i in self.Y])
        self.ESS = sum([(y_i - self.Y_mn)**2 for y_i in self.Y_pr])
        self.R = self.ESS/self.TSS
        self.RSS = sum([(self.Y[i] - self.Y_pr[i])**2 for i in range(len(self.Y))])
    def regression_coefs(self):
        xtx = np.dot(np.transpose(self.X),self.X)
        xty = np.dot(np.transpose(self.X),self.Y)
        coefs = np.linalg.solve(xtx, xty)
        self.rgc = coefs
        return
    def F_test_for_significance(self, alpha):
        F = (self.R**2/(len(self.X[0]) - 1))/((1 - self.R**2)/(self.df + 1))
        print("------------------------------------------")
        print(F)
        print(fcv(len(self.X[0]) - 1, self.df + 1, alpha))
        return True if F > fcv(len(self.X[0]) - 1, self.df + 1,alpha) else False

    def F_test_for_variable(self, j):
        X_T = np.transpose(X)
        r_model = multi_regression(X_T[:j] + X_T[j+1:], Y)
        F = ((r_model.RSS - self.RSS)/(r_model.df - self.df))/(self.RSS/self.df)
        return F