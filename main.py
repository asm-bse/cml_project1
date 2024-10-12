import pandas as pd
from lr_model import lr_results
from random_forest import rf_results
from gradient_boosting import gbr_results

# Словарь для всех результатов
results = {
    'Linear Regression': lr_results,
    'Random Forest': rf_results,
    'Gradient Boosting': gbr_results
}

# Создание списка для табличного отображения результатов
rows = []
for model_name, result in results.items():
    rows.append({
        'Model': model_name,
        'MSE': result['MSE'],
        'R² Score': result['R²']
    })

# Преобразование списка в DataFrame для вывода в виде таблицы
results_df = pd.DataFrame(rows)

# Вывод таблицы с результатами моделей
print("Результаты моделей:")
print(results_df)

# Поиск лучшей модели по MSE
best_model_name = results_df.loc[results_df['MSE'].idxmin(), 'Model']
best_result = results[best_model_name]

print(f"\nЛучшая модель: {best_model_name}")
print(f"  MSE: {best_result['MSE']}")
print(f"  R²: {best_result['R²']}")

# Сравнение предсказанных и реальных значений для лучшей модели
predicted_vs_actual = pd.DataFrame({
    'Predicted': best_result['y_pred'],
    'Actual': best_result['y_test']
})
print("\nСравнение предсказанных и реальных значений (первые 5 записей):")
print(predicted_vs_actual.head())


import joblib

# Сохранение лучшей модели
best_model_name = min(results, key=lambda x: results[x]['MSE'])
best_model = results[best_model_name]['model']
joblib.dump(best_model, 'best_model.pkl')  # Сохранение модели в файл

