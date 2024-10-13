from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scaling import scaling

X_train_scaled, X_test_scaled, y_train, y_test = scaling()
# Случайный лес
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Прогнозирование
y_pred_rf = rf_model.predict(X_test_scaled)

# Оценка модели
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Результаты
rf_results = {
    'model': rf_model,
    'MSE': mse_rf,
    'R²': r2_rf,
    'y_pred': y_pred_rf,
    'y_test': y_test
}
