
def apply_model():
    import pandas as pd
    import joblib
    from sklearn.preprocessing import StandardScaler
    from loguru import logger

    logger.info("applying best_model.pkl")
    neigbourhood = 'Gràcia'
    best_model = joblib.load('best_model.pkl')

    
    df_test = pd.read_csv('test.csv')
    df_test = df_test[df_test['neighborhood'] == neigbourhood]

    #cleaning data
    df_test_cleaned = df_test[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'is_furnished', 'has_pool', 'num_crimes', 'has_ac', 'accepts_pets']]
    df_test_cleaned = pd.get_dummies(df_test_cleaned, drop_first=True)
    df_test_cleaned = df_test_cleaned.fillna(df_test_cleaned.mean()) 

    # Scaling
    scaler = StandardScaler()
    df_test_scaled = scaler.fit_transform(df_test_cleaned)

    #applying model
    predictions = best_model.predict(df_test_scaled)
    df_test['predicted_price'] = predictions

    # output
    #print(df_test[['num_rooms', 'num_baths', 'square_meters', 'predicted_price']])
    df_test.to_csv('test_with_predictions.csv', index=False)
    logger.info("test_with_predictions.csv generated")

if __name__ == '__main__':
    apply_model()
