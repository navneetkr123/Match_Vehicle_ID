#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('Speed_dataset.csv')
dataset.drop(['timestamp','state', 'driver_id'], axis=1, inplace=True)
X=dataset.iloc[:, 0:7].values
y=dataset.iloc[:, 7:8].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X_1=LabelEncoder()
X[:, 5]=label_encoder_X_1.fit_transform(X[:, 5])
onehotencoder=OneHotEncoder(categorical_features=[5])
X=onehotencoder.fit_transform(X).toarray()
X=X[:, 1:]

label_encoder_y_1=LabelEncoder()
y[:, 0]=label_encoder_y_1.fit_transform(y[:, 0])
y=np.asarray(y)
y=np.ravel(y)

#Splitting the dataset into test and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier =RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Important_Features (Training set)')
plt.xlabel('Important_Features')
plt.ylabel('Vehicle_ID')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Important_Features (Test set)')
plt.xlabel('Important_Features')
plt.ylabel('Vehicle_ID')
plt.show()