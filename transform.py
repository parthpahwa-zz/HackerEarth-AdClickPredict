import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Browsers: {nan, 'Safari', 'Opera', 'Chrome', 'Mozilla Firefox', 'Edge', 'IE','Google Chrome', 'Internet Explorer', 'Firefox', 'InternetExplorer', 'Mozilla'}
# devid : {nan, 'Desktop', 'Tablet', 'Mobile'}

# ID             object
# datetime       datetime64[ns]
# siteid         float64
# offerid        int64
# category       int64
# merchant       int64
# countrycode    object
# browserid      object
# devid          object
# click          int64
strCols = ['countrycode', 'browserid', 'devid']


def fillNan(df):
	for col in df.columns.values:
		df[col].fillna(0, inplace = True)


# convitring timestamp into finer granular structure
def convDateTime(df):
	df['hour'] = df.datetime.dt.hour
	df['day'] = df.datetime.dt.weekday
	df['min'] = df.datetime.dt.minute


# Browserid contains multiple lables for a unique browser, hence cleaning
def cleanBrowser(df):
	df.loc[df.browserid.isin(['Google Chrome', 'Chrome']), 'browserid'] = 'GOOG'
	df.loc[df.browserid.isin(['Mozilla Firefox', 'Mozilla', 'Firefox']), 'browserid'] = 'MOZ'
	df.loc[df.browserid.isin(['Internet Explorer', 'InternetExplorer', 'IE']), 'browserid'] = 'MS'
	df.browserid.fillna('None', inplace=True)


def genFeatures(df):
	# number of ads with parituclar merchants in a specific country
	temp = df.groupby(['merchant', 'countrycode'])['ID'].count().reset_index()
	df = df.merge(temp, 'left', ['merchant', 'countrycode'], suffixes=('', '_merchantCountryTotal'))
	
	# number of ads with paritcular devices running a specific browser
	temp = df.groupby(['devid', 'browserid'])['ID'].count().reset_index()
	df = df.merge(temp, 'left', ['devid', 'browserid'], suffixes=('', '_devBrowserTotal'))

	# number of ads active on a day at some time grouped by site & offer  
	for col in ['siteid', 'offerid']:
		temp = df.groupby([col, 'day', 'hour'])['ID'].count().reset_index()
		df = df.merge(temp, 'left', [col, 'day', 'hour'], suffixes=('', '_' + col + 'HourTotal'))
	
	# number of ads active on a day at some time with granularity minutes grouped by category & merchant
	for col in ['category', 'merchant']:
		temp = df.groupby([col, 'day', 'hour', 'min'])['ID'].count().reset_index()
		df = df.merge(temp, 'left', [col, 'day', 'hour', 'min'], suffixes=('', '_' + col + 'DayHourMinCount'))

	# number of ads active on a day at some time grouped by siteid & merchant
	temp = df.groupby(['siteid', 'merchant', 'day', 'hour'])['ID'].count().reset_index()
	df = df.merge(temp, 'left', ['siteid', 'merchant', 'day', 'hour'], suffixes=('', '_siteMerchantHourTotal'))

	# number of ads active on a day at some time grouped by siteid & offerid
	temp = df.groupby(['siteid', 'offerid', 'day', 'hour'])['ID'].count().reset_index()
	df = df.merge(temp, 'left', ['siteid', 'offerid','day', 'hour'], suffixes=('', '_siteOfferHourTotal'))

	# number of ads on a particular site, on a specific browser being used on some device
	temp = df.groupby(['devid', 'browserid', 'siteid'])['ID'].count().reset_index()
	df = df.merge(temp, 'left', ['devid', 'browserid', 'siteid'], suffixes=('', '_devBrowserSiteTotal'))

	return df


# Encode catagorical values
def encode(df):
	encdr = LabelEncoder()
	df.devid.fillna('None', inplace=True)
	for col in strCols:
		df[col] = encdr.fit_transform(df[col])


# Convert the float columns to int to save space as all columns except ID and datetime are integer values 
def convFloatToInt(df):
	for col in df.columns.values:
		if col not in ['ID', 'datetime']:
			df[col] = df[col].astype(int)
	return df


# The Dataset is not well distributed hence undersample the rows with click = 0 to create approx 50/50 distribution
# Click == 0 : 11700596
# Click == 1 : 437214
def performUnderSampling(df):
	df = pd.concat([df.loc[df.click == 0].sample(438000), df.loc[df.click == 1]]).reset_index(drop=True)
	df = df.sample(frac=1).reset_index(drop=True)
	return df


def transform(df_train, name='train'):
	print "Filling empty values"
	fillNan(df_train)
	
	print "Converting Day Time"
	convDateTime(df_train)
	
	print "Relabeling Browsers"
	cleanBrowser(df_train)
	
	print "Generating Features"
	df_train = genFeatures(df_train)

	print "Started Encoding"
	encode(df_train)
	
	print "Conv FLoat to Int"
	df_train = convFloatToInt(df_train)
	
	if not name  == "test":
		print "Train UnderSampling:"
		# Perform undersampling at end to reduce loss of information during feature engineering
		df_train = performUnderSampling(df_train)
	
	return df_train

# transform train.csv to transformedTrain.csv 
df = pd.read_csv("train.csv", parse_dates=['datetime'], dtype={'siteid': np.float, 'offerid': np.float, 'category': np.float, 'merchant': np.float, 'click': np.float})
df = transform(df)
df.to_csv("transformedTrain.csv", index=False)

# transform test.csv to transformedTest.csv
df = pd.read_csv("test.csv", parse_dates=['datetime'], dtype={'siteid': np.float, 'offerid': np.float, 'category': np.float, 'merchant': np.float, 'click': np.float})
df = transform(df, 'test')
df.to_csv("transformedTest.csv", index=False)
 