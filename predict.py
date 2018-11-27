# Importing required libraries:
import sklearn as sk 
import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import preprocessing, model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Create dataframes:
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Plot data for visualization:
#sns.barplot(x = "Embarked", y = "Survived",  data = train_data);
#plt.show()
#print(train_data.Age.describe())

# Cleaning dataframe:

def FeatureDrop(df):
	return df.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 1)

def QAge(df):
	df.Age = df.Age.fillna(-0.5)
	bins = (-1,0,5,11,20,30,60,81)
	bin_names = ['Unknown','Toddler','Child','Youth','Young Adult','Adult','Senior']
	categories = pd.cut(df.Age, bins, labels = bin_names)
	df.Age = categories
	return df

def QSibSp(df):
	df.SibSp = df.SibSp.fillna(0)
	bins = (-0.5,0.5,15)
	bin_names = ['No', 'Yes']
	categories = pd.cut(df.SibSp, bins, labels = bin_names)
	df.SibSp = categories
	return df 

def QParch(df):
	df.Parch = df.Parch.fillna(0)
	bins = (-0.5,0.5,15)
	bin_names = ['No', 'Yes']
	categories = pd.cut(df.Parch, bins, labels = bin_names)
	df.Parch = categories
	return df

def Clean_features(df):
	df = FeatureDrop(df)
	df = QParch(df)
	df = QAge(df)
	df = QSibSp(df)
	return df

train_data = Clean_features(train_data)
test_data = Clean_features(test_data)

# Make all the features numeric:

def make_numeric(train_data, test_data):
	features = ['Sex', 'Age', 'SibSp', 'Parch']
	combined_data = pd.concat([train_data[features], test_data[features]])
	for feature in features:
		le = preprocessing.LabelEncoder()
		le = le.fit(combined_data[feature])
		train_data[feature] = le.transform(train_data[feature])
		test_data[feature] = le.transform(test_data[feature])

	return train_data, test_data

train_data, test_data = make_numeric(train_data, test_data)

# Split the data into training and testing sets:

x_data = train_data.drop(['PassengerId', 'Survived'], axis = 1)
y_data = train_data['Survived']

test_size = 0.20

x_train, y_train, x_test, y_test = train_test_split(x_data, y_data, test_size = test_size, random_state = 23)

# Build the classifier:

clf = RandomForestClassifier()

parameters = {'n_estimators': [4,6,9], 'max_features': ['log2','sqrt','auto'], 'criterion':['entropy','gini'], 'max_depth':[2,3,5,10], 'min_samples_split':[2,3,5],'min_samples_leaf':[1,5,8]}

scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
grid_obj = grid_obj.fit(x_train, y_train)

clf = grid_obj.best_estimator_
clf.fit(x_train, y_train)

# Prediction:

prediction = clf.predict(x_test)
print(accuracy_score(y_test, prediction))



#print(train_data.sample(3))