''' Bike Sharing Dataset Python File

    Dataset: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset 

    Dataset Description:
    
    - instant: record index
    - dteday : date
    - season : season (1:springer, 2:summer, 3:fall, 4:winter)
    - yr : year (0: 2011, 1:2012)
    - mnth : month ( 1 to 12)
    - hr : hour (0 to 23)
    - holiday : weather day is holiday or not (extracted from [Web Link])
    - weekday : day of the week
    - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
    + weathersit : 
    - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
    - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
    - temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
    - atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
    - hum: Normalized humidity. The values are divided to 100 (max)
    - windspeed: Normalized wind speed. The values are divided to 67 (max)
    - casual: count of casual users
    - registered: count of registered users
    - cnt: count of total rental bikes including both casual and registered
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import metrics
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

def load_df(filepath):
    """load the dataset into the system using pandas.read_csv function
    
    Parameters:
    filepath: Path to the dataset
    
    Returns:
    df: Generated dataframe"""
    df=pd.read_csv(filepath)
    return df

def data_prep(df):
    '''Preparation of the dataset into train and test data to used in the models
    
        Parameters:
        df: dataframe name

        Returns:
        X_train,X_test,y_train,y_test: train and test dataset

    '''
    
    #check for missing values:
    assert pd.notnull(df).all().all()
    
    #feature generation from date:
    '''We already have features like Month, Year and dayofweek as column of the dataframe.'''
    df['day'] = pd.DatetimeIndex(df['dteday']).day
    
    # Normalization of the right skewed distribution target variable:
    print('Normalizing count column of df')
    df['cnt'] = np.log(df['cnt'])
    print('Normalization done')
    
    # Drop leakage columns as they generate too good to be true prediction score for the models:
    df = df.drop(['dteday', 'atemp', 'casual', 'registered'],axis=1)
    
    #Divide dataframe into trainset and testset
    X = df.drop(['cnt'], axis=1)
    y = df['cnt']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    
    return X_train, X_test, y_train, y_test

def generate_model(X_train, X_test, y_train, y_test, regressor):
    """ Function to generate machine learning model and return score, mean absolute error and predicted output
        
        Parameters:
        X_train, X_test, y_train, y_test: train test split
        regressor: the regression algorithm to be used
        
        Returns:
        score: machine learning score of the model
        mae: mean absolute error
        y_preds: prediction output
    """
    
    # The parameters of the regressors are generated after applying GridSearchCV
    
    regressor.fit(X = X_train, y = np.log1p(y_train))
    y_preds = regressor.predict(X_test)
    score = regressor.score(X_test,np.log1p(y_test))
    mae = metrics.mean_absolute_error(y_test, np.exp(y_preds))
    
    return score, mae, y_preds

def test_model(filepath):
    """ load dataset, build feature set, and do learning
        
        Parameters:
        fn: file name of dataset
        features: a list of list, each of which is a feature list for different models
        type: str for indicating feature set
        
        Returns:
        predictions and feature-engineered dataset are saved to files
    """
    df = load_df(filepath)
    
    X_train, X_test, y_train, y_test = data_prep(df)
    
    for i, regressor in enumerate((
        RandomForestRegressor(n_estimators=100, max_depth=20),
        GradientBoostingRegressor(n_estimators=150, max_depth=10, min_samples_leaf=20, learning_rate=0.1),
        SVR(kernel='rbf', C=50))):
        score, mae, y_preds = generate_model(X_train, X_test, y_train, y_test, regressor)
        rname = str(regressor).split('(')[0]
        print(rname, score, mae)
        results = pd.DataFrame({'hr': X_test.loc[:,'hr'], 'cnt': y_test, 'prediction': np.exp(y_preds)})
        results.to_csv('...\results.csv', index = False, columns=['hr', 'cnt', 'prediction'])
    

if __name__ == "__main__":
    test_model(filepath='...\hour.csv')