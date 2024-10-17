# EX3 Implementation of Linear Regression Using Gradient Descent
## DATE:

## AIM:
To write a program to implement the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize w and 𝑏
2.Compute the cost function.
3.Compute the gradients for w and b
4.Update the parameters and repeat until convergence.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: monisha.L
RegisterNumber: 2305001019 
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
  x=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(x.shape[1]).reshape(-1,1)
  for i in range(num_iters):
    predictions=x.dot(theta).reshape(-1,1)
    error=(predictions-y).reshape(-1,1)
    gradient=x.T.dot(error)
    theta=theta-learning_rate*gradient
  return theta
data=pd.read_csv('/content/50_Startups-2.csv',header=None)
X = (data.iloc[1:, :-2].values)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
theta = linear_regression(X1_Scaled, Y1_Scaled)
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![image](https://github.com/user-attachments/assets/26aefce6-e7aa-4c07-866e-8b5be0800d89)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
