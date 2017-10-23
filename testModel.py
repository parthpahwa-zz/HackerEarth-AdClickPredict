import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.externals import joblib

df = pd.read_csv("transformedTest.csv", parse_dates=['datetime'], dtype={'siteid': np.float, 'offerid': np.float, 'category': np.float,
				'merchant': np.float, 'click': np.float})

def predXGB(df):
	X_all = df.drop(['ID', 'datetime'], axis=1)
	gbm = joblib.load("adClick.joblib.dat")
	predictions = gbm.predict(X_all)
	df_final = pd.DataFrame(columns = ['ID', 'click'])
	df_final['ID'] = df['ID']
	df_final['click'] = predictions
	df_final.to_csv('submission.csv', index=False)

predXGB(df)