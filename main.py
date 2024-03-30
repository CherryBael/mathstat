# импорт библиотек
import pandas as pd
import matplotlib.pyplot as plt
# для нахождения квантилей распределения Стьюдента
import scipy

# класс, содержащий датасет и методы для работы с парной линейной регрессией
class pair_reg:
    #подсчет среднего произведений
    def double_mean(self,X, Y):
        q = 0
        s = 0
        # тут где-то должна быть проверка на одинаковую длину столбцов
        for i in range(len(X)):
            s += X[i]*Y[i]
            q += 1
        return s / q
    # возвращает значение квантиля распределния Стьюдента
    def t_quant(self, n, alpha):
        return scipy.stats.t.ppf((1 + alpha)/2, n)

    def __init__(self, data):
        assert(len(data.transpose()) == 2) # проверка на то, что датасете 2 столбца
        self.X, self.Y = data.iloc[:, 0], data.iloc[:, 1]
        assert(len(self.X) == len(self.Y)) # проверка на равенство длины массивов, потом нужно будет сделать обработчик исключений
        self.X_mean, self.Y_mean = self.X.mean(), self.Y.mean()
        self.Xsq_mean = self.double_mean(self.X, self.X) 
        self.Ysq_mean = self.double_mean(self.Y, self.Y)
        self.XY_mean = self.double_mean(self.X, self.Y)
        self.b_1 = (self.XY_mean - self.X_mean*self.Y_mean)/(self.Xsq_mean - self.X_mean**2)
        self.b_0 = self.Y_mean - self.b_1 * self.X_mean
        self.RSS = sum([((self.Y[i] - self.X[i] * self.b_1 - self.b_0) ** 2) for i in range(len(self.X))])
        self.eval_sigma = self.RSS / len(self.X)
        self.Ssq_b1 = self.eval_sigma / sum([(self.X[i] - self.X_mean)**2 for i in range(len(self.X))])
        self.Ssq_b0 = self.Xsq_mean*self.Ssq_b1
        return 
    def trusted_intervals(self, alpha):
        return [self.b_0 - self.t_quant(len(self.X) - 2, alpha) * self.Ssq_b0 ** (1/2), self.b_0 + self.t_quant(len(self.X) - 2, alpha) * self.Ssq_b0 ** (1/2)], [self.b_1 - self.t_quant(len(self.X) - 2, alpha) * self.Ssq_b1 ** (1/2), self.b_1 + self.t_quant(len(self.X) - 2, alpha) * self.Ssq_b1 ** (1/2)]
    def is_valuable(self, alpha):
        tinX, tinY = self.trusted_intervals(alpha)
        fl1 = False if tinX[0] < 0 and tinX[1] > 0 else True
        fl2 = False if tinY[0] < 0 and tinY[1] > 0 else True
        return fl1, fl2
# Возможно имеет смысл сделать ленивый подсчет всех полей класса

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
rg = pair_reg(data)
print("b_0 =",rg.b_0)
print("b_1 =",rg.b_1)
tX, tY = rg.trusted_intervals(0.05)

print(f"b_0 lies from {tX[0]} to {tX[1]}")
sig0, sig1 = rg.is_valuable(0.05)
print("b_0 is significant" if sig0 else "b_0 is insignificant")
print(f"b_1 lies from {tY[0]} to {tY[1]}")
print("b_1 is significant" if sig1 else "b_1 is insignificant")
plot_regression_line(rg.X, rg.Y, [rg.b_0, rg.b_1])
