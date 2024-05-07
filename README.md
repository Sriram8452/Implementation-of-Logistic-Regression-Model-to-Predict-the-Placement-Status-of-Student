# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. by using google colab .
2. Upload the dataset .
3. check null , duplicated values using .isnull() and .duplicated() function respectively.
4. Import LabelEncoder and encode the dataset.
5. Import LogisticRegression from sklearn and apply the model on the dataset.
6. Predict the values of array
7. Calculate the accuracy, confusion and classification report by importing the required modules
from sklearn.
8. Apply new unknown values

## Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: Sriram G

RegisterNumber:  212222230149

*/

```
import pandas as pd
data = pd.read_csv('/content/Placement_Data (1).csv')
data.head()
data1=data.copy()

data1=data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
print(data1)

x=data1.iloc[:,:-1]
print(x)

y=data1["status"]
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```




## Output:

## Read data:

![image](https://github.com/Sriram8452/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708032/55c0d4e2-ce98-49d1-87a2-1a19a6ce21d7)

## Droping unwanted coloumn:

![image](https://github.com/Sriram8452/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708032/453bb407-8743-454e-9c1f-d1df58e3864b)

## Presence of null value


![image](https://github.com/Sriram8452/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708032/4451580f-0932-43a1-863b-8c77b3a5d600)

## Duplicated value

![image](https://github.com/Sriram8452/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708032/505db50d-cc56-47a1-9fc6-adea0e212ce9)

## Data encoding


![image](https://github.com/Sriram8452/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708032/49b2af74-c018-46ba-a1d9-d0f80bd9b30d)

## X data


![image](https://github.com/Sriram8452/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708032/e3b42d83-5905-40b7-80b0-df87650fe585)

## Y data


![image](https://github.com/Sriram8452/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708032/845dfe95-a1c2-45b3-acbc-fb5118c53d87)

## Predicted values

![image](https://github.com/Sriram8452/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708032/d2fb809c-8a4d-455b-867c-c622ed47e805)

## Accuracy value


![image](https://github.com/Sriram8452/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708032/a57fcbc6-8bdc-4fe9-9cd6-068f048dad4d)

## Confusion matrix


![image](https://github.com/Sriram8452/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708032/ea9b09b3-674b-436a-8521-4d72289950b9)

## Classification report


![image](https://github.com/Sriram8452/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708032/80ddf74a-3387-4260-a9a5-4283f54196e6)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

