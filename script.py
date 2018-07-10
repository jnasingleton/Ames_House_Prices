# This script creates a ton of features, then trains an ensemble of XGBoost regressor, Lasso regressor, ElasticNet and KernelRidge.
#
# It borrows ideas from lots of other people's scripts, including:
# https://www.kaggle.com/humananalog/house-prices-advanced-regression-techniques/xgboost-lasso
# https://www.kaggle.com/klyusba/house-prices-advanced-regression-techniques/lasso-model-for-regression-problem/notebook
# https://www.kaggle.com/juliencs/house-prices-advanced-regression-techniques/a-study-on-regression-applied-to-the-ames-dataset/
# https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

import datetime

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, BayesianRidge, LassoLarsIC, LassoCV
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel

#https://stackoverflow.com/questions/43327020/xgboostlibrarynotfound-cannot-find-xgboost-library-in-the-candidate-path-did-y?rq=1
#import xgboost as xgb
import lightgbm as lgb

import numpy as np
import pandas as pd

#%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
from scipy.stats import norm, skew

start_time = datetime.datetime.now()

def create_binary_features_equality(df, create_binary_features_list):
	# Create new binary featurea based on existing featurea and criteria.
	# Possibly remove existing feature once a binary feature has been created.

	for feature_tocreate, feature_todestroy, feature_criteria  in create_binary_features_list:
		df[feature_tocreate] = (df[feature_todestroy] == feature_criteria) * 1.0
		#Drop feature_todestory

	return df

def create_binary_features_isnull(df, create_binary_features_list):
	# Create new binary featurea based on existing featurea and criteria.
	# Possibly remove existing feature once a binary feature has been created.

	for feature_tocreate, feature_todestroy  in create_binary_features_list:
		df[feature_tocreate] = 1 - ((df[feature_todestroy].isnull()) * 1.0)
		#Drop feature_todestory

	return df

def create_binary_features(df):
	# Create new binary features based on existing features.
	# Possibly remove existing feature once binary feature has been created.

    create_binary_features_list = [['HasCentralAir','CentralAir','Y'],
    								['IsRegularLotShape','LotShape','Reg'],
    								['IsLandLevel','LandContour','Lvl'],
    								['IsLandSlopeGentle','LandSlope','Gtl'],
    								['IsElectricalSBrkr','Electrical','SBrkr'],
    								['IsGarageDetached','GarageType','Detchd'],
    								['IsPavedDrive','PavedDrive','Y'],
    								['HasShed','MiscFeature','Shed'],
    								['Has2ndFloor','2ndFlrSF',0],
    								['HasMasVnr','MasVnrArea',0],
    								['HasWoodDeck','WoodDeckSF',0],
    								['HasOpenPorch','OpenPorchSF',0],
    								['HasEnclosedPorch','EnclosedPorch',0],
    								['Has3SsnPorch','3SsnPorch',0],
    								['HasScreenPorch','ScreenPorch',0],
    								['BoughtOffPlan','SaleCondition','Partial']
    								] 
    df = create_binary_features_equality(df, create_binary_features_list)

    create_binary_features_list = [['HasBasement','BsmtQual'],
    								['HasGarage','GarageQual'],
    								['HasFireplace','FireplaceQu'],
    								['HasFence','Fence']
    								] 
    df = create_binary_features_isnull(df, create_binary_features_list)

    # If YearRemodAdd != YearBuilt, then a remodeling took place at some point.
    df['Remodeled'] = (df['YearRemodAdd'] != df['YearBuilt']) * 1

    # Did a remodeling happen in the year the house was sold?
    df['RecentRemodel'] = (df['YearRemodAdd'] == df['YrSold']) * 1
    
    # Was this house sold in the year it was built?
    df['RecentBuild'] = (df['YearBuilt'] == df['YrSold']) * 1

    # Is the MSSubClass a '1946 & NEWER' type
    df['NewerDwelling'] = df['MSSubClass'].map(
        {20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 
        	70: 0, 75: 0, 80: 0, 85: 0, 90: 0, 
        	120: 1, 150: 0, 160: 1, 180: 0, 190: 0}).astype(int)   

    # Was the sale possibly priced down?
    df['SaleCondition_PriceDown'] = df.SaleCondition.replace(
        {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})

    return df

###  

### TARGET VARIABLE

def display_prob_distribution(train_df, feature):
	# Plots the feature, the feature's fitted normal distribution, and the kde

	# Plot the feature, the feature's fitted normal distribution, and the kde (Gaussian Kernel Density Estimate)
	sns.distplot(train_df[feature], kde=True, fit=norm);

	# Get the fitted parameters used by the function
	(mu, sigma) = norm.fit(train_df[feature])
	print(feature + ': ' + 'mu = {:.2f} and sigma = {:.2f}'.format(mu, sigma))

	# Set up the plot attributes
	plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
	plt.ylabel('Frequency')
	plt.title(feature + ' distribution')

	# Create the figure for plt
	fig = plt.figure()

	# Show the QQ plot related to plt
	res = stats.probplot(train_df[feature], plot=plt)

	# Show fig and res related to plt
	plt.show()

###		

### OUTLIERS

def remove_outliers(train_df):
	# Remove outliers from a dataframe

	# There are a few houses with more than 4000 sq ft living area that are outliers, 
	# so we drop them from the training data. 
	# (There is also one in the test set but we obviously can't drop that one).
	train_df.drop(train_df[train_df['GrLivArea'] > 4000].index, inplace=True)

	return train_df

###

### MISSING DATA

def find_na_features(train_df, test_df):
	# Displays all NA values (as a percentage) for all_df

	ntrain = train_df.shape[0]
	ntest = test_df.shape[0]

	all_df = pd.concat((train_df, test_df)).reset_index(drop=True)

	# Remove the target variable from the dataframe
	all_df.drop(['SalePrice'], axis=1, inplace=True)

	all_na_df = (all_df.isnull().sum() / len(all_df)) * 100
	all_na_df = all_na_df.drop(all_na_df[all_na_df == 0].index)
	all_na_df = all_na_df.sort_values(ascending=False)

	missing_data = pd.DataFrame({'Missing Ratio' :all_na_df})

	#display_results = False
	display_results = True
	if display_results:
		print(missing_data)

	return train_df, test_df

def fill_na_features_0(all_df, fill_na_features_0_list):
	# Fill NA values in specified features with 0s

	for feature in fill_na_features_0_list:
		all_df[feature].fillna(0, inplace=True)

	return all_df

def fill_na_features_none(all_df, fill_na_features_none_list):
	# Fill NA values in specified features with 0s

	for feature in fill_na_features_none_list:
		all_df[feature].fillna('None', inplace=True)

	return all_df

def fill_na_features_median(all_df, train_df, fill_na_features_median_list):
	# Fill NA values in specified features with the median, grouped by another feature
	# We use the median of train_df so we don't use information from test_df to influence train_df

	for feature_tofill, feature_togroup in fill_na_features_median_list:

		#for key, group in all_df[feature_tofill].groupby(all_df[feature_togroup]):
		for key, group in train_df[feature_tofill].groupby(train_df[feature_togroup]):
			idx = (all_df[feature_togroup] == key) & (all_df[feature_tofill].isnull())
			all_df.loc[idx, feature_tofill] = group.median()

	return all_df

def fill_na_features_mode_simple(all_df, train_df, fill_na_features_mode_simple_list):
	# Fill NA values in specified features with the mode, grouped on the same column
	# We use the mode of train_df so we don't use information from test_df to influence train_df

	for feature in fill_na_features_mode_simple_list:
		#all_df[feature] = all_df[feature].fillna(all_df[feature].mode()[0])
		all_df[feature] = all_df[feature].fillna(train_df[feature].mode()[0])

	return all_df

def fill_na_features(train_df, test_df):
	# Fill NA values in features

	ntrain = train_df.shape[0]
	ntest = test_df.shape[0]

	all_df = pd.concat((train_df, test_df)).reset_index(drop=True)

	# Fill NA values with 0
	fill_na_features_0_list = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath',
								'GarageArea','GarageCars',
								'MasVnrArea',
								'PoolArea']
	all_df = fill_na_features_0(all_df, fill_na_features_0_list)

	# Fill NA values with 'None'
	fill_na_features_none_list = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','MasVnrType','MSSubClass',
									'GarageType','GarageFinish','GarageQual','GarageCond',
									'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'] 
	all_df = fill_na_features_none(all_df, fill_na_features_none_list)

	# Fill NA values with the mode, grouped on the same feature
	fill_na_features_mode_simple_list = ['MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType']
	all_df = fill_na_features_mode_simple(all_df, train_df, fill_na_features_mode_simple_list)

	# Fill NA values with the median, grouped on a different feature
	fill_na_features_median_list = [['LotFrontage','Neighborhood']] 
	all_df = fill_na_features_median(all_df, train_df, fill_na_features_median_list)

	# Other Fill NA value processes
	all_df['Functional'] = all_df['Functional'].fillna('Typ')
	all_df['GarageYrBlt'] = all_df['GarageYrBlt'].fillna(all_df['YearBuilt'])

	# Remove features if unused
	#Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . 
	#Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling and can be safely removed.
	all_df = all_df.drop(['Utilities'], axis=1)

	train_df = all_df[:ntrain]
	test_df = all_df[ntrain:]

	return train_df, test_df

###

### FEATURE ENGINEERING

def convert_quality_features_fromlist(all_df, convert_quality_features_list, convert_quality_features_dict):
	# Convert specific text quality features into integers, using a provided mapping
	
	for feature in convert_quality_features_list:
		all_df[feature] = all_df[feature].map(convert_quality_features_dict).astype(int)

	return all_df

def convert_quality_features(all_df):
	# Convert text quality features into integers
    # Quality measurements are stored as text.
    # We can convert them to integers where a higher number represents a higher quality.

    convert_quality_features_list = ['ExterQual','ExterCond',
    									'BsmtQual','BsmtCond',
    									'HeatingQC','KitchenQual','FireplaceQu',
    									'GarageQual','GarageCond',
    									'PoolQC']
    convert_quality_features_dict = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}									
    all_df = convert_quality_features_fromlist(all_df, convert_quality_features_list, convert_quality_features_dict)

    convert_quality_features_list = ['BsmtExposure']
    convert_quality_features_dict = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}								
    all_df = convert_quality_features_fromlist(all_df, convert_quality_features_list, convert_quality_features_dict)

    convert_quality_features_list = ['BsmtFinType1','BsmtFinType2']
    convert_quality_features_dict = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}								
    all_df = convert_quality_features_fromlist(all_df, convert_quality_features_list, convert_quality_features_dict)

    convert_quality_features_list = ['Functional']
    convert_quality_features_dict = {'None': 0, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}					
    all_df = convert_quality_features_fromlist(all_df, convert_quality_features_list, convert_quality_features_dict)

    convert_quality_features_list = ['GarageFinish']
    convert_quality_features_dict = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}				
    all_df = convert_quality_features_fromlist(all_df, convert_quality_features_list, convert_quality_features_dict)   

    convert_quality_features_list = ['Fence']
    convert_quality_features_dict = {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}				
    all_df = convert_quality_features_fromlist(all_df, convert_quality_features_list, convert_quality_features_dict) 

    return all_df

def create_new_features(all_df, train_df):
	# Create new features based on existing features

	area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    				'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
    				'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea' ]
	all_df['TotalArea'] = all_df[area_cols].sum(axis=1)

	all_df['TotalArea1st2nd'] = all_df['1stFlrSF'] + all_df['2ndFlrSF']
    
	all_df['YearsSinceBuilt'] = 2010 - all_df['YearBuilt']

	all_df['MonthsSinceSold'] = 2010*12 - ((all_df['YrSold']*12) - 12 + all_df['MoSold'])

	all_df['YearsSinceRemodel'] = all_df['YrSold'] - all_df['YearRemodAdd']

	# Add a MonthQual to weight months by total count of sales by month
	# We use the count of train_df so we don't use information from test_df to influence train_df
	feature_dict = train_df['MoSold'].groupby(train_df['MoSold']).count().to_dict()
	all_df['MonthQual'] = all_df['MoSold'].map(feature_dict)  

	# Add a YearQual to weight months by total count of sales by year
	# We use the count of train_df so we don't use information from test_df to influence train_df
	feature_dict = train_df['YrSold'].groupby(train_df['YrSold']).count().to_dict()
	all_df['YearQual'] = all_df['YrSold'].map(feature_dict)  

	# Add a NeighborhoodQual to weight neighborhoods by median SalesPrice by Neighborhood
	# We use the median of train_df so we don't use information from test_df to influence train_df
	feature_dict = train_df['SalePrice'].groupby(train_df['Neighborhood']).median().to_dict()
	all_df['NeighborhoodQual'] = all_df['Neighborhood'].map(feature_dict)   

	return all_df

def convert_features_to_str(all_df):
	# Convert numerical features to str if they don't have an ordering

	all_df['MSSubClass'] = all_df['MSSubClass'].apply(str)
	# YrSold has an ordering, newer/higher years should have higher prices
	#all_df['YrSold'] = all_df['YrSold'].astype(str)
	all_df['MoSold'] = all_df['MoSold'].astype(str)

	return all_df

def find_skewed_features(all_df):
	# Check the skew of all numerical features

	numeric_feats = all_df.dtypes[all_df.dtypes != "object"].index
	skewed_feats = all_df[numeric_feats].apply(lambda x: skew(x.dropna()))
	skewed_feats = skewed_feats.sort_values(ascending=False)

	skewness_limit = 1
	skewed_feats = skewed_feats [abs(skewed_feats ) > skewness_limit].dropna()

	display_results = False
	#display_results = True
	if display_results:
		print("There are {} skewed numerical features".format(skewed_feats .shape[0]))
		print(skewed_feats)


	return skewed_feats

def find_removable_features(train_df_temp, test_df_temp):

	train_df_temp = pd.get_dummies(train_df_temp)
	test_df_temp = pd.get_dummies(test_df_temp)

	train_df_columnlist = list(train_df_temp.columns.values)
	test_df_columnlist = list(test_df_temp.columns.values)

	# https://www.geeksforgeeks.org/python-find-missing-additional-values-two-lists/
	# Missing from train
	# If a feature exists in test but not train, having this feature in test does not impact prediction
	train_df_columnlist_missing = list(set(test_df_columnlist).difference(train_df_columnlist))
	# Missing from test
	# If a feature exists in train but not test, having this feature in train will not help prediction
	test_df_columnlist_missing = list(set(train_df_columnlist).difference(test_df_columnlist))

	df_columnlist_toremove = train_df_columnlist_missing + test_df_columnlist_missing

	return df_columnlist_toremove

# Polynomial features could be created on the features that have the greatest impact/importances
def feature_engineering(train_df, test_df):
	# Create/convert features

	ntrain = train_df.shape[0]
	ntest = test_df.shape[0]

	all_df = pd.concat((train_df, test_df)).reset_index(drop=True)

	# Convert categorical features to int if they do have an ordering
	all_df = convert_quality_features(all_df)

	# Create new features based on existing features
	all_df = create_new_features(all_df, train_df)

	# Convert numerical features to str if they don't have an ordering
	all_df = convert_features_to_str(all_df)

	# We could simplify some categorical features to reduce size of feature space (ie. not used in get_dummies)

	skewed_feats = find_skewed_features(all_df)

	# Apply log1p (applies log(1+x) to all elements), which makes the distribution more 'normal'
	# We know the minimum of all values in all features is 0, so we are able to use log1p or log if needed
	skewed_feats_index = skewed_feats.index
	#--
	all_df[skewed_feats_index] = np.log1p(all_df[skewed_feats_index])
	#--
	#from scipy.special import boxcox1p
	#lam = 0.15
	##lam can be determined through http://onlinestatbook.com/2/transformations/box-cox.html
	#for feat in skewed_feats_index:
	#    all_df[feat] = boxcox1p(all_df[feat], lam)   
	#--

	#skewed_feats = find_skewed_features(all_df)

	# Find a list of features that should be removed because they will not help prediction
	train_df_temp = all_df[:ntrain]
	test_df_temp= all_df[ntrain:]
	df_columnlist_toremove = find_removable_features(train_df_temp, test_df_temp)

	# Convert test features to numerical features and remove features not needed
	all_df = pd.get_dummies(all_df)
	all_df.drop(df_columnlist_toremove, axis=1, inplace=True)

	train_df = all_df[:ntrain]
	test_df = all_df[ntrain:]

	return train_df, test_df
	
### BASE MODELLING

def rmsle_cv(model):
	# Used this specific metric due to it being the metric used for the Kaggle Competition

	n_folds = 5

	#kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_df.values)
	cv = KFold(n_folds, shuffle=True, random_state=42)
	#rmse= np.sqrt(-cross_val_score(model, train_df.values, train_y, scoring="neg_mean_squared_error", cv = kf))
	rmse= np.sqrt(-cross_val_score(model, train_df.values, train_y, scoring="neg_mean_squared_error", cv = cv))
	return(rmse)

def linearregression_model():

	regressor = LinearRegression()
	scaler = RobustScaler()

	linearregression_pipeline = Pipeline(
		[
		('feature_selection', SelectFromModel(regressor)), 
		('scaler', scaler), 	
		('classification', regressor)
		])

	score = rmsle_cv(linearregression_pipeline)
	print("Linear Regression score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

	return linearregression_pipeline, score

def lassocv_model():

	#lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005))
	regressor = LassoCV()
	scaler = RobustScaler()

	LassoCV_pipeline = Pipeline(
		[
		('feature_selection', SelectFromModel(regressor)), 
		('scaler', scaler), 	
		('classification', regressor)
		])

	score = rmsle_cv(LassoCV_pipeline)
	print("LassoCV score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

	return LassoCV_pipeline, score

def enet_model():

	#ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
	regressor = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
	scaler = RobustScaler()

	elasticnet_pipeline = Pipeline(
		[
		('feature_selection', SelectFromModel(regressor)), 
		('scaler', scaler), 	
		('classification', regressor)
		])

	score = rmsle_cv(elasticnet_pipeline)
	print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

	return elasticnet_pipeline, score

def krr_model():

	regressor = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

	score = rmsle_cv(regressor)
	print("Kernel Ridge Regression score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

	return regressor, score

def gboost_model():

	# Huber loss is a combination of ls (least squares) and lad (least absolute deviation); it has moderate insensitity to outliers
	regressor = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

	score = rmsle_cv(regressor)
	print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

	return regressor, score

def xgboost_model():

	regressor = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

	score = rmsle_cv(regressor)
	print("X Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

	return regressor, score

def lgbm_model():

	regressor = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

	score = rmsle_cv(regressor)
	print("Light GBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

	return regressor, score

### ENSEMBLE MODELLING

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models
        
    # Define clones of the original models and fit them
    def fit(self, X, y):

        # Define clones of the original models
        self.models_ = [clone(x) for x in self.models]
        
        # Fit cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    # Predict for the cloned models and average them
    def predict(self, X):

    	# Predict for each model and combine to a single 2d array
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])

        # Take the mean/average of the predictions
        prediction = np.mean(predictions, axis=1)

        return prediction 

###

# Load the data
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')

### OUTLIERS

# We remove outliers at the start so that the lengths of target_variable/index_variable/train/test/ dataframes all match

train_df = remove_outliers(train_df)

### ID VARIABLE

target_variable = 'Id'

# Save the ID column
train_ID_df = train_df[target_variable]
test_ID_df = train_df[target_variable]

# Note: We cannot drop and rows from train_df or test_df, unless we also updated train_ID_df/test_ID_df at the same indices

# Drop the ID column since it's unnecessary for the prediction process.
train_df.drop(target_variable, axis = 1, inplace = True)
test_df.drop(target_variable, axis = 1, inplace = True)

### TARGET VARIABLE

target_variable = 'SalePrice'

#display_prob_distribution(train_df, target_variable)
# Apply log1p (applies log(1+x) to all elements), which makes the distribution more 'normal'
# We know the minimum of all values in all features is 0, so we are able to use log1p or log if needed
train_df[target_variable] = np.log1p(train_df[target_variable])
#display_prob_distribution(train_df, target_variable)

train_y = train_df[target_variable].values

### MISSING DATA

#train_df, test_df = find_na_features(train_df, test_df)
train_df, test_df = fill_na_features(train_df, test_df)	
#train_df, test_df = find_na_features(train_df, test_df)	

### LOGIC TESTS

# Need to add logic tests: (ie if a Garage does not exist, yet GarageArea > 0)

### FEATURE ENGINEERING

train_df, test_df = feature_engineering(train_df, test_df)		

### NORMALIZATION

# All features are now numerical because we used get_dummies during feature_engineering
# We can now apply normalization to all features
# Normalization will be done as part of the pipeline process

### FEATURE SELECTION

# Feature selection will be done as part of the pipeline process

### CROSS VALIDATION

# Cross Validation will be done as part of the pipeline process

### BASE MODELLING

linearregression_pipeline, score = linearregression_model()

lassocv_pipeline, score = lassocv_model()

enet_pipeline, score = enet_model()

krr_regressor, score = krr_model()

gboost_regressor, score = gboost_model()

#xgboost_regressor, score = xgboost_model()

lgbm_regressor, score = lgbm_model()

### ENSEMBLE MODELLING

averaged_models = AveragingModels(models = (lassocv_pipeline, enet_pipeline, krr_regressor, gboost_regressor, lgbm_regressor))
#averaged_models = AveragingModels(models = (gboost_regressor, lgbm_regressor))

score = rmsle_cv(averaged_models)
print("Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

### NEXT STEPS / IMPROVEMENTS

# Use bayesian optimization to determine cooeficients, try: https://github.com/fmfn/BayesianOptimization
# Investigate other additional base models
# Hyperparameter search/test 
# Investigate why linear regression score is 0
