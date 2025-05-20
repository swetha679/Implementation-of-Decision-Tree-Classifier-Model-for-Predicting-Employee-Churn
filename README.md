# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipment Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision
3. Fit the data in the model
4. Find the accuracy score

## Program:
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: B R SWETHA NIVASINI 

RegisterNumber: 212224040345

```
import pandas as pd
data = pd.read_csv("C:\\Users\\admin\\OneDrive\\Desktop\\Folders\\ML\\DATASET-20250226\\Employee.csv")
data.head()
```

![image](https://github.com/user-attachments/assets/4a93d1d1-a9b8-458d-8da9-9ddd0d58cbd7)

```
data.info()
data.isnull().sum()
data['left'].value_counts()
```
![image](https://github.com/user-attachments/assets/82dffe3d-2531-419d-a6c5-bce25e628736)

```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])
data.head()
```
![image](https://github.com/user-attachments/assets/62a67c73-7633-4981-bdbb-796fdb7337e8)

```
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
```
![image](https://github.com/user-attachments/assets/9aabb266-32fa-4927-ac9b-87b89d1d4e4a)

```
y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy
```
![image](https://github.com/user-attachments/assets/092f0e0d-2f1c-41ea-8ffc-f64037a9dd23)

```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
![image](https://github.com/user-attachments/assets/2d9f6941-339b-45f3-b186-7519446bea8b)

```
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
![image](https://github.com/user-attachments/assets/35289f4c-7227-48fa-a9aa-59c41b83202f)










# Result:
Thus, the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using Python programming.
