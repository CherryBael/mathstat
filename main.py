#импортируем модули
from lib.pair_regression import pair_reg
from lib.utils import prepare_data, plot_regression_line
alpha = 0.05
data = prepare_data()
print(data)
rg = pair_reg(data)
print("b_0 =",rg.b_0)
print("b_1 =",rg.b_1)
tX, tY = rg.trusted_intervals(alpha)
print(f"b_0 lies from {tX[0]} to {tX[1]}")
sig0, sig1 = rg.is_valuable(alpha)
print(f"b_0 is significant with alpha = {alpha}" if sig0 else f"b_0 is insignificant with alpha = {alpha}")
print(f"b_1 lies from {tY[0]} to {tY[1]}")
print(f"b_1 is significant with alpha = {alpha}" if sig1 else f"b_1 is insignificant with alpha = {alpha}")
rg.intervals_plot(alpha)
# старый вывод графика
#plot_regression_line(rg.X, rg.Y, [rg.b_0, rg.b_1])
