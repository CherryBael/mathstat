import pandas as pd
import matplotlib.pyplot as plt
# для нахождения квантилей распределения Стьюдента
import scipy
#подготовка данных
def prepare_data():
    data = pd.read_csv('dataset/student_exam_data.csv', delimiter=',')
    # убираем лишние данные
    data = data.transpose()[0:2].transpose()
    # перемешиваем данные
    data = data.sample(frac=1).reset_index(drop=True)
    # отрезаем 400 строк -- нашу выборку
    data = data[0:400]
    return data

#подсчет среднего произведений
def double_mean(a, b):
    q = 0
    s = 0
    # тут где-то должна быть проверка на одинаковую длину столбцов
    for i in range(len(a)):
        s += a[i]*b[i]
        q += 1
    return s / q

def find_RSS_coef(dataframe, name_col_x, name_col_y):
    # находим средние по столбцам
    x_mean = dataframe[name_col_x].mean()
    y_mean = dataframe[name_col_y].mean()
    xy_mean = double_mean(dataframe[name_col_x], dataframe[name_col_y])
    xsq_mean = double_mean(dataframe[name_col_x], dataframe[name_col_x])
    # считаем коэффициенты
    b_1 = (xy_mean - x_mean*y_mean)/(xsq_mean - x_mean**2)
    b_0 = y_mean - b_1 * x_mean
    return b_0, b_1

# строим и выводим график
def plot_regression_line(x, y, b):
    plt.scatter(x, y, color = "m",
    marker = "o", s = 30)
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color = "g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
# возвращает значение квантиля распределния Стьюдента
def t_quant(n, alpha):
    return scipy.stats.t.ppf((1 + alpha)/2, n)
# находит сумму квадратов ошибок
def find_RSS(dataframe, name_col_x, name_col_y):
    b_0, b_1 = find_RSS_coef(data, "Previous Exam Score","Study Hours")
    rss = 0
    for i in range(len(data[name_col_y])):
        rss += (data[name_col_y][i] - data[name_col_x][i] * b_1 - b_0) ** 2
    return rss
def find_eval_sigma_s(dataframe, name_col_x, name_col_y):
    return find_RSS(dataframe, name_col_x, name_col_y) / (len(dataframe[name_col_x]) - 2)

def sq_b_1(dataframe, name_col_x, name_col_y):
    b_0, b_1 = find_RSS_coef(dataframe, name_col_x, name_col_y)
    xsq_mean = double_mean(dataframe[name_col_x], dataframe[name_col_x])
    x_mean = dataframe[name_col_x].mean()
    return find_eval_sigma_s(dataframe, name_col_x, name_col_y)/(len(dataframe[name_col_x]) * (xsq_mean - x_mean**2))
def sq_b_0(dataframe, name_col_x, name_col_y):
    xsq_mean = double_mean(dataframe[name_col_x], dataframe[name_col_x])
    return sq_b_1(dataframe, name_col_x, name_col_y) * xsq_mean
def trusted_intervals(dataframe, name_col_x, name_col_y, alpha):
    b_0, b_1 = find_RSS_coef(dataframe, name_col_x, name_col_y)[0], find_RSS_coef(dataframe, name_col_x, name_col_y)[1]
    q = t_quant(len(dataframe[name_col_x]) - 2, alpha)
    sqb0 = sq_b_0(dataframe, name_col_x, name_col_y)
    sqb1 = sq_b_1(dataframe, name_col_x, name_col_y)
    return [[b_0 - q * sqb0 ** (1/2), b_0 + q * sqb0 ** (1/2)],[b_1 - q * sqb0 ** (1/2), b_1 + q * sqb0 ** (1/2)]]
def is_valuable(dataframe, name_col_x, name_col_y, alpha):
    tins = trusted_intervals(dataframe, name_col_x, name_col_y, alpha)
    fl1 = False if tins[0][0] < 0 and tins[0][1] > 0 else True
    fl2 = False if tins[1][0] < 0 and tins[1][1] > 0 else True
    return fl1, fl2


data = prepare_data()
#print(data)
# найдем коэффициенты
koefs = find_RSS_coef(data, "Previous Exam Score","Study Hours")
print("b_0 =",koefs[0])
print("b_1 =",koefs[1])
ans =trusted_intervals(data, "Previous Exam Score","Study Hours", 0.05)
print(f"b_0 lies from {ans[0][0]} to {ans[0][1]}")
sig0, sig1 = is_valuable(data, "Previous Exam Score","Study Hours", 0.05)
print("b_0 is significant" if sig0 else "b_0 is insignificant")
print(f"b_1 lies from {ans[1][0]} to {ans[1][1]}")
print("b_1 is significant" if sig1 else "b_1 is insignificant")
#plot_regression_line(data["Previous Exam Score"], data["Study Hours"], ans)

