# pandas и numpy
import pandas as pd
import numpy as np
# для нахождения квантилей распределения Стьюдента
import scipy
# визуализация
import matplotlib.pyplot as plt
import seaborn as sns

#подготовка данных
def prepare_data_for_student_exam():
    data = pd.read_csv('dataset/student_exam_data.csv', delimiter=',')
    # убираем лишние данные
    data = data.transpose()[0:2].transpose()
    # перемешиваем данные
    data = data.sample(frac=1).reset_index(drop=True)
    # отрезаем 400 строк -- нашу выборку
    data = data[0:400]
    # меняем местами столбцы (мы хотим поменять местами X и Y)
    data.iloc[:, [0,1]] = data.iloc[:, [1, 0]]
    data.columns = [data.columns[1], data.columns[0]]
    return data
def prepare_data_for_heart_failure():
    data = pd.read_csv('dataset/heart_failure_clinical_records.csv', delimiter=',')
    # перемешиваем данные
    data = data.sample(frac=1).reset_index(drop=True)
    # отрезаем 2000 строк -- нашу выборку
    data = data[0:2000]
    tmpX = data.drop(columns = ["ejection_fraction"])
    X = tmpX.values
    Xnames = tmpX.columns
    tmpY = data["ejection_fraction"]
    Y = tmpY.values
    Yname = tmpY.name
    return X, Y, Xnames, Yname
# строим и выводим график
def plot_regression_line(x, y, b, ):
    plt.scatter(x, y, color = "m",
    marker = "o", s = 30)
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color = "g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def is_ortogonal(A,B):
    ans = np.dot(np.transpose(A), B)
    zz = np.zeros((len(np.transpose(A)), len(np.transpose(B))))
    return np.array_equal(ans, zz)
def t_quant(n, alpha):
        return scipy.stats.t.ppf((1 + alpha)/2, n)
def f_critical_value(n, m, alpha):
    return scipy.stats.f.ppf(1 - alpha, n, m)
