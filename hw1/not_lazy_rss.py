import pandas as pd
import matplotlib.pyplot as plt

#подготовка данных
def prepare_data():
    data = pd.read_csv('student_exam_data.csv', delimiter=',')
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
    x_mean = data[name_col_x].mean()
    y_mean = data[name_col_y].mean()
    xy_mean = double_mean(data[name_col_x], data[name_col_y])
    xsq_mean = double_mean(data[name_col_x], data[name_col_x])
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

data = prepare_data()
print(data)
ans = find_RSS_coef(data, "Previous Exam Score","Study Hours")
print("b_0 =",ans[0])
print("b_1 =",ans[1])
plot_regression_line(data["Previous Exam Score"], data["Study Hours"], ans)

