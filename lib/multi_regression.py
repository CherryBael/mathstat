# pandas и numpy
import pandas as pd
import numpy as np
# функция рассчета критических значений
from lib.utils import f_critical_value as fcv, t_quant
# считаем коэффициенты линейной регрессии
class multi_regression:
    class cache:
        Cor_Var = None
        Cov = None
        SE = None
        TSS = None
        ESS = None
        RSS = None
        R = None
    def __init__(self, X, Y):
        try:
            assert(len(X) == len(Y)) # проверка на равенство длины массивов, потом нужно будет сделать обработчик исключений
        except:
            print("Количество наблюдений в X и Y не совпадает")
            print("Невозможно продолжить выполнение программы")
            exit(1)
        self.X = X
        self.Y = Y
        self.regression_coefs()       # создает поле rgc -- коэфициенты линейной регрессии
        self.df = len(self.Y) - len(self.X[0]) - 1  # количество степеней свободы
        self.Y_pr = np.dot(self.X, self.rgc)
        self.Y_mn = self.Y.mean()
        self.e = self.Y - self.Y_pr
    @property
    def Cor_Var(self):
        if self.cache.Cor_Var is None: 
            self.cache.Cor_Var = np.dot(np.transpose(self.e),self.e)/self.df
        return self.cache.Cor_Var
    @property
    def Cov(self):
        if self.cache.Cov is None:
            self.cache.Cov = self.Cor_Var * np.linalg.pinv(np.dot(np.transpose(self.X),self.X))
        return self.cache.Cov
    @property
    def SE(self):
        if self.cache.SE is None:
            self.cache.SE = [self.Cov[i][i] for i in range(len(self.Cov))]
        return self.cache.SE
    @property
    def TSS(self):
        if self.cache.TSS is None:
            self.cache.TSS = sum([(y_i - self.Y_mn)**2 for y_i in self.Y])
        return self.cache.TSS
    @property
    def ESS(self):
        if self.cache.ESS is None:
            self.cache.ESS = sum([(y_i - self.Y_mn)**2 for y_i in self.Y_pr])
        return self.cache.ESS
    @property
    def RSS(self):
        if self.cache.RSS is None:
            self.cache.RSS = sum([(self.Y[i] - self.Y_pr[i])**2 for i in range(len(self.Y))])
        return self.cache.RSS
    @property
    def R(self):
        if self.cache.R is None:
            self.cache.R = self.ESS/self.TSS
        return self.cache.R








    def regression_coefs(self):
        xtx = np.dot(np.transpose(self.X),self.X)
        xty = np.dot(np.transpose(self.X),self.Y)
        coefs = np.linalg.solve(xtx, xty)
        self.rgc = coefs
        return
    def trust_interval(self,i, alpha):
        return(self.rgc[i] - self.SE[i] * t_quant(self.df, alpha), self.rgc[i] + self.SE[i] * t_quant(self.df, alpha))

    def F_test_for_significance(self, alpha):
        F = (self.R**2/(len(self.X[0]) - 1))/((1 - self.R**2)/(self.df + 1))
        #print("------------------------------------------")
        #print(F)
        #print(fcv(len(self.X[0]) - 1, self.df + 1, alpha))
        return True if F > fcv(len(self.X[0]) - 1, self.df + 1,alpha) else False

    def F_test_for_variable(self, j):
        X_T = np.transpose(X)
        r_model = multi_regression(X_T[:j] + X_T[j+1:], Y)
        F = ((r_model.RSS - self.RSS)/(r_model.df - self.df))/(self.RSS/self.df)
        return F