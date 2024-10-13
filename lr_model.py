
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scaling import scaling

X_train_scaled, X_test_scaled, y_train, y_test = scaling()
# Линейная регрессия
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Прогнозирование
y_pred_lr = lr_model.predict(X_test_scaled)

# Оценка модели
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Результаты
lr_results = {
    'model': lr_model,
    'MSE': mse_lr,
    'R²': r2_lr,
    'y_pred': y_pred_lr,
    'y_test': y_test
}
