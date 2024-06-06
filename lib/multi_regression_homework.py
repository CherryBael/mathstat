from lib.multi_regression import multi_regression
from lib.utils import prepare_data_for_heart_failure, is_ortogonal as iso
def multi_regression_homework():
    #X = np.array([[1,2,-1,3], [-1,1,0,0], [0,2,1,0], [0,0,1,-1],[1,-1,2,1]])
    #Y = np.transpose(np.array([1,2,3,4,5]))
    X, Y, Xnames, Yname = prepare_data_for_heart_failure()
    alpha = 0.01       # для F теста
    beta = 0.05        # для доверительных интервалов
    round_accuracy = 3 # количество знаков после запятой в выводе в stdout 
    rm = multi_regression(X, Y)
    for i in range(len(Xnames)):
        print(f"Коэффициент регрессии между {Yname} и {Xnames[i]} равен {round(rm.rgc[i], round_accuracy)}")
        tleft, tright = [round(x, round_accuracy) for x in rm.trust_interval(i, beta)]
        print(f"Принадлежит интервалу [{tleft},{tright}] с точностью {(1 - beta) * 100}%")
    print(f"Модель проходит F-тест на значимость с точностью {(1 - alpha) * 100}%" if rm.F_test_for_significance(alpha) else f"Модель не проходит F-тест на значимость с точностью {(1 - alpha) * 100}%")
    return