from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scaling import scaling

X_train_scaled, X_test_scaled, y_train, y_test = scaling()
# Градиентный бустинг
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_model.fit(X_train_scaled, y_train)

# Прогнозирование
y_pred_gbr = gbr_model.predict(X_test_scaled)

# Оценка модели
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)

# Результаты
gbr_results = {
    'model': gbr_model,
    'MSE': mse_gbr,
    'R²': r2_gbr,
    'y_pred': y_pred_gbr,
    'y_test': y_test
}
