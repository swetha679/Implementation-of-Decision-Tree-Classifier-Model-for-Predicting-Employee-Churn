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
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: B R SWETHA NIVASINI 
Register Number:  212224040345

```
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x. head () #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```



#  Output:
![decision tree classifier model](sam.png)

![Screenshot 2025-04-19 163414](https://github.com/user-attachments/assets/dda07a10-7147-4956-b078-fae22624ebe7)

![Screenshot 2025-04-19 163423](https://github.com/user-attachments/assets/f192f20d-f2a5-49e0-b46a-582e38dc1ffe)

![Screenshot 2025-04-19 163431](https://github.com/user-attachments/assets/5ed16b40-d311-46e6-b7d7-80d4458df36c)

![Screenshot 2025-04-19 163439](https://github.com/user-attachments/assets/784e6d71-0423-4459-8ee7-f082f6cdf3e6)

![Screenshot 2025-04-19 163450](https://github.com/user-attachments/assets/809822f2-3aee-440d-980d-ff0865e1b86f)

![Screenshot 2025-04-19 163503](https://github.com/user-attachments/assets/8a2fca99-a35b-46a0-942d-9f1d421492ec)

![Screenshot 2025-04-19 163512](https://github.com/user-attachments/assets/fa12bd3c-ac9c-4928-897c-7cf4b383a1e9)

![Screenshot 2025-04-19 163526](https://github.com/user-attachments/assets/1e2e4cf8-2903-459f-89eb-b8b1288f2f32)










# Result:
Thus, the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using Python programming.
