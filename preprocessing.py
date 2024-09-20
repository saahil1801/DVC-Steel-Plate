import pandas as pd

def preprocess_data(df):
    """
    Preprocess the data by engineering features and dropping unnecessary columns.
    """
    try:
        df['X'] = df['X_Maximum'] - df['X_Minimum']
        df['Y'] = df['Y_Maximum'] - df['Y_Minimum']
        df['Luminosity'] = df['Maximum_of_Luminosity'] - df['Minimum_of_Luminosity']
        df['Area_Perimeter_Ratio'] = df['Pixels_Areas'] / (df['X_Perimeter'] + df['Y_Perimeter'])
        
        # Drop original columns
        df = df.drop([
            'X_Maximum', 'X_Minimum', 'Y_Maximum', 'Y_Minimum',
            'Maximum_of_Luminosity', 'Minimum_of_Luminosity',
            'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter'
        ], axis=1)
    except KeyError as e:
        print(f"Missing column: {e}")
    
    return df

def load_data(train_path, test_path):
    """
    Load the training and testing data from specified paths.
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    return df_train, df_test

def save_data(df_train, df_test, train_output_path, test_output_path):
    """
    Save preprocessed train and test data to specified paths.
    """
    df_train.to_csv(train_output_path, index=False)
    df_test.to_csv(test_output_path, index=False)

if __name__ == "__main__":
    # Paths to input data
    train_data_path = 'playground-series-s4e3/train.csv'
    test_data_path = 'playground-series-s4e3/test.csv'

    # Paths to output data
    preprocessed_train_path = 'preprocessed_train.csv'
    preprocessed_test_path = 'preprocessed_test.csv'

    # Load and preprocess the data
    df_train, df_test = load_data(train_data_path, test_data_path)
    df_train = preprocess_data(df_train)
    df_test = preprocess_data(df_test)

    # Save preprocessed data to files
    save_data(df_train, df_test, preprocessed_train_path, preprocessed_test_path)
