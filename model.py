import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

df = pd.read_csv("transformedTrain.csv", parse_dates=['datetime'], dtype={'siteid': np.float, 'offerid': np.float, 'category': np.float,
				'merchant': np.float, 'click': np.float})

def predXGB(df):
	y_all = df['click']
	X_all = df.drop(['ID', 'datetime', 'click'], axis=1)
	gbm = xgb.XGBClassifier(max_depth= 9, n_estimators=700, learning_rate=0.005).fit(X_all, y_all)

	#Save the model for later use 
	joblib.dump(gbm, "adClick.joblib.dat")
	
def findFeatureImportance(df):
	'''
	Feature Importance:

	0.0680543 siteid
	0.0354232 offerid
	0.0354796 category
	0.0590151 merchant
	0.0330259 countrycode
	0.0431086 browserid
	0.0285275 devid
	0.0587613 hour
	0.0110556 day
	0.0242124 min
	0.033731 ID_merchantCountryTotal
	0.0429111 ID_devBrowserTotal
	0.114166 ID_siteidHourTotal
	0.0621739 ID_offeridHourTotal
	0.0275263 ID_siteMerchantHourTotal
	0.0787715 ID_siteOfferHourTotal
	0.11686 ID_devBrowserSiteTotal
	0.0696054 ID_categoryDayHourMinCount
	0.0575909 ID_merchantDayHourMinCount
	'''
	
	y_all = df['click']
	X_all = df.drop(['ID', 'datetime', 'click'], axis=1)
	
	X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=11)
	clf= xgb.XGBClassifier(max_depth = 9, n_estimators=700, learning_rate=0.005, silent=False).fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test,y_test)], eval_metric='auc', verbose=True )
	
	predictions = clf.predict(X_test)
	print(accuracy_score(y_test, predictions))

	for i in range(0,len(clf.feature_importances_)):
		print clf.feature_importances_[i], X_train.columns.values[i]

predXGB(df)
