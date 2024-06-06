import time
# pandas и numpy
import pandas as pd
import numpy as np
from lib.multi_regression import multi_regression
from lib.utils import prepare_data_for_heart_failure

def multi_regression_homework():
    #X = np.array([[1,2,-1,3], [-1,1,0,0], [0,2,1,0], [0,0,1,-1],[1,-1,2,1]])
    #Y = np.transpose(np.array([1,2,3,4,5]))
    X, Y, Xnames, Yname = prepare_data_for_heart_failure()
    alpha = 0.05
    ts = time.time()
    rm = multi_regression(X, Y)
    for i in range(len(Xnames)):
        print(f"Коэффициент регрессии между {Yname} и {Xnames[i]} равен {rm.rgc[i]}")
    print(f"Модель проходит F-тест на значимость с точностью {(1 - alpha) * 100}%" if rm.F_test_for_significance(alpha) else f"Модель не проходит F-тест на значимость с точностью {(1 - alpha) * 100}%")
    #print(time.time() - ts)
    return
