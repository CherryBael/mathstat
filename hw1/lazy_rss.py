import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
def estimate_coef(x, y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return (b_0, b_1)

def ret_col(mtrx, ncol):
    ans = []
    for i in range(len(mtrx)):
        ans.append(mtrx[i][ncol])
    return ans

def plot_regression_line(x, y, b):
    plt.scatter(x, y, color = "m",
    marker = "o", s = 30)
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color = "g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

a = np.genfromtxt("dataset/student_exam_data.csv", delimiter=",")
x = ret_col(a, 1)
y = ret_col(a, 0)
ind = [i for i in range(1, len(x))]
ind = random.sample(ind, 400)
print(ind)
x_new = []
y_new = []
for i in range(len(ind)):
    x_new.append(x[ind[i]])
    y_new.append(y[ind[i]])
x = np.array(x_new)
y = np.array(y_new)
b_0, b_1 = estimate_coef(x, y)
print(b_0, b_1)
plot_regression_line(x, y, [b_0, b_1])
