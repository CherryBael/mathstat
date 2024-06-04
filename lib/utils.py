# pandas и numpy
import pandas as pd
import numpy as np
# для нахождения квантилей распределения Стьюдента
import scipy
# визуализация
import matplotlib.pyplot as plt
import seaborn as sns
# pandas и numpy
import pandas as pd
import numpy as np
# для нахождения квантилей распределения Стьюдента
import scipy
# визуализация
import matplotlib.pyplot as plt
import seaborn as sns

#подготовка данных
def prepare_data():
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

# строим и выводим график
def plot_regression_line(x, y, b, ):
    plt.scatter(x, y, color = "m",
    marker = "o", s = 30)
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color = "g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
