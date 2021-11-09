import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import xgboost as xg
sns.set()
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(filename='development_logs.txt',
                    filemode='a',
                    format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

logging.warning("Reading Dataset...")
raw_data = pd.read_csv('concrete_data.csv')
data = raw_data
logging.warning("Successfully completed")
logging.warning("Raw data shape: (1030,9)")
logging.warning("Removing extra space...")
data=raw_data.rename({"fine_aggregate ":"fine_aggregate"},axis=1)
logging.warning("Analysing outliers...")
logging.warning("Dropping fly_ash and water attributes...")
data.drop(["fly_ash"],axis=1,inplace=True)
data.drop(["water"],axis=1,inplace=True)
logging.warning("Successfully completed")
logging.warning("Exporting the preprocessed file...")
data.to_csv('out.csv', index=False) 
logging.warning("Successfully completed")