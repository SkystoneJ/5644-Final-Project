from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

X_train = pd.read_csv('train3X.csv')
y_train = pd.read_csv('train3y.csv')
X_test = pd.read_csv('test3X.csv')
y_test = pd.read_csv('test3y.csv')
# 初始化随机森林回归器
regressor = RandomForestRegressor(n_estimators=400, random_state=42, max_depth=20, min_samples_split=3,
                                  min_samples_leaf=1, n_jobs=-1)

# 训练模型
regressor.fit(X_train, y_train)

# 预测测试集
y_pred = regressor.predict(X_test)

# 计算 MSE 作为性能指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')


# model = LinearRegression()
#
# # 训练模型
# model.fit(X_train, y_train)
#
# # 预测测试集
# y_pred = model.predict(X_test)
#
# # 计算评估指标
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
# print(f'R² Score: {r2}')
