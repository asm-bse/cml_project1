
def compare_models():
    import pandas as pd
    from lr_model import lr_results
    from random_forest import rf_results
    from gradient_boosting import gbr_results
    from loguru import logger
    import joblib

    # Dict for rsults
    results = {
        'Linear Regression': lr_results,
        'Random Forest': rf_results,
        'Gradient Boosting': gbr_results
    }

    # List for results as a table
    rows = []
    for model_name, result in results.items():
        rows.append({
            'Model': model_name,
            'MSE': result['MSE'],
            'R² Score': result['R²']
        })

    results_df = pd.DataFrame(rows)

    # Results of models
    logger.info("Results of models:")
    print(results_df)

    # Поиск лучшей модели по MSE
    best_model_name = results_df.loc[results_df['MSE'].idxmin(), 'Model']
    best_result = results[best_model_name]

    print('\n')
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"  MSE: {best_result['MSE']}")
    logger.info(f"  R²: {best_result['R²']}")

    #Сomraining predicted and actual values for the best model:"
    predicted_vs_actual = pd.DataFrame({
        'Predicted': best_result['y_pred'],
        'Actual': best_result['y_test']
    })
    #print("\nСomraining predicted and actual values for the best model:")
    #print(predicted_vs_actual.head())

    #Exporting the best model
    best_model_name = min(results, key=lambda x: results[x]['MSE'])
    best_model = results[best_model_name]['model']
    joblib.dump(best_model, 'best_model.pkl') 
    logger.info("best_model.pkl generated")


if __name__ == '__main__':
    compare_models()
    