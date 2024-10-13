
def clean_data():
    import pandas as pd
    from loguru import logger

    neigbourhood = 'Gràcia'
    num_rooms_trigger = 42 # got this looking at data set row 39, 42 rooms and more seem to be outliers

    df = pd.read_csv('train.csv')

    #df = df[['num_rooms', 'num_baths', 'square_meters', 'orientation', 'year_built', 'door', 'is_furnished', 'has_pool', 'neighborhood', 'num_crimes', 'has_ac', 'accepts_pets', 'num_supermarkets']]

    df = df[df['neighborhood'] == neigbourhood]

    #removing num_supermarkets and orientation bc there are too many missing values and they are not useful. 
    #door and neighborhood are useless as well

    df = df[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'is_furnished', 'has_pool', 'num_crimes', 'has_ac', 'accepts_pets', 'price']]
    initial_rows = len(df)
    logger.info(f'initial_rows: {initial_rows}')
    #removing rows with irrational values
    df = df[df['num_rooms'] < num_rooms_trigger]
    df = df[df['square_meters'] > 0]
    remaining_rows = len(df)
    logger.info(f'remaining_rows ater excluding oultiers: {remaining_rows}')
    #checking for missing values
    na_values = df.isna().sum()
    #print(f"Missing values:\n{na_values}")
    #df['missing_values_count'] = df.isna().sum(axis=1)

    ### lets remove all the missing values and see if the dataset we have is big enough to be useful

    df_cleaned = df.dropna() #df_cleaned = df.fillna(df.mean()) wont change final resuilts

    remaining_rows = len(df_cleaned)
    logger.info(f"deleted_rows: {initial_rows - remaining_rows}")
    logger.info(f'remaining_rows ofter processing nan values: {remaining_rows}')

    #print(df_cleaned.describe())
    #print(df_cleaned.describe(include='object'))

    #all the values seem reasonable, without oultiers
    df_cleaned = df_cleaned[['num_rooms', 'num_baths', 'square_meters', 'year_built', 'is_furnished', 'has_pool', 'num_crimes', 'has_ac', 'accepts_pets', 'price']]
    df_cleaned.to_csv('train_cleaned.csv', index=False)
    return(df_cleaned)

if __name__ == '__main__':
    clean_data()
