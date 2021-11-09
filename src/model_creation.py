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

logging.warning("Reading preprocessed dataset...")
data = pd.read_csv("out.csv")
logging.warning("Declaring variables...")
y = data['concrete_compressive_strength']
x=data.drop(['concrete_compressive_strength'],axis=1)
logging.warning("Successfully completed")
logging.warning("Splitting the dataset...")
train_X, test_X, train_y, test_y = train_test_split(x, y,
                      test_size = 0.3, random_state = 123)
logging.warning("Successfully completed")
logging.warning("Import libraries for ML algorithms...")
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
logging.warning("Successfully completed")
logging.warning("Analysing results on 3 different algorithms...")
Model_Names=["Linear Regression","Random Forest","Decision Tree"]
Scores=[]
lr=LinearRegression()
lr.fit(train_X,train_y)
y_head=lr.predict(test_X)
Scores.append(r2_score(test_y,y_head))

rf=RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(train_X,train_y)
y_head=rf.predict(test_X)
Scores.append(r2_score(test_y,y_head))

dt=DecisionTreeRegressor()
dt.fit(train_X,train_y)
y_head=dt.predict(test_X)
Scores.append(r2_score(test_y,y_head))
logging.warning("Successfully completed")

logging.warning("Initialising random forest regressor...")
regr = RandomForestRegressor()
logging.warning("Fitting train-test data...")
regr.fit(train_X,train_y)
logging.warning("Pickling the model...")
import pickle
pickle_out = open("strength.pkl","wb")
pickle.dump(regr, pickle_out)
pickle_out.close()
logging.warning("Successfully completed")