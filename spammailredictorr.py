import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import mean_squared_error, f1_score

data=pd.read_csv('C:\\Users\\yadav\\OneDrive\\Desktop\\spambase.csv')
print(data.shape)
#creating data for LinearRegression
# Accessing the 58th column by index (57th index in zero-based indexing)
column_58 = data.iloc[:, 57]
Y=column_58
X = data.iloc[:, np.r_[:56, 57:]]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2,stratify=Y)
#creating data for SVM
scaler=StandardScaler()
scaler.fit(X)
stddata=scaler.transform(X)
XX_train,XX_test,YY_train,YY_test=train_test_split(stddata,Y,test_size=0.2,random_state=2,stratify=Y)
#create models
model=LinearRegression()
model2=svm.SVC(kernel='linear')
#training the models
model.fit(X_train,Y_train)
model2.fit(XX_train,YY_train)
#testing the models
p1=model.predict(X_test)
p2=model2.predict(XX_test)

# Calculate mean squared error for linear regression
mse = mean_squared_error(p1, Y_test)
print("The mean error for Linear regression",mse)

# Calculate f1 score for SvM
f1 = f1_score(p2, Y_test)
print("Accuracy score for SVM :", f1)
