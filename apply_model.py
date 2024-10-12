import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from cleaning_data import neigbourhood
# Шаг 1: Загрузка модели
best_model = joblib.load('best_model.pkl')

# Шаг 2: Загрузка нового набора данных (test.csv)
df_test = pd.read_csv('test.csv')
df_test = df_test[df_test['neighborhood'] == neigbourhood]
# Шаг 3: Очистка данных аналогично тренировочным данным
df_test_cleaned = df_test[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'is_furnished', 'has_pool', 'num_crimes', 'has_ac', 'accepts_pets']]
df_test_cleaned = df_test_cleaned.fillna(df_test_cleaned.mean())  # Заполнение пропусков средними значениями

# Шаг 4: Преобразование категориальных данных (кодирование)
df_train_cleaned = pd.read_csv('train_cleaned.csv')  # Например, сохранённые тренировочные данные после обработки
df_test_cleaned = pd.get_dummies(df_test_cleaned, drop_first=True)

# Применение тех же столбцов, которые были в тренировочном наборе
df_test_cleaned = df_test_cleaned.reindex(columns=df_train_cleaned.columns, fill_value=0)

# Шаг 5: Масштабирование данных
scaler = StandardScaler()
df_test_scaled = scaler.fit_transform(df_test_cleaned)

# Шаг 6: Применение модели для предсказания
predictions = best_model.predict(df_test_scaled)

# Шаг 7: Вывод предсказанных значений
df_test['predicted_price'] = predictions
print(df_test[['num_rooms', 'num_baths', 'square_meters', 'predicted_price']])

# Можно сохранить результат в новый CSV файл
df_test.to_csv('test_with_predictions.csv', index=False)
