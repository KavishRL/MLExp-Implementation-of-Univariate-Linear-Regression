# MLExp-Implementation-of-Univariate-Linear-Regression
AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
Get the independent variable X and dependent variable Y.
Calculate the mean of the X -values and the mean of the Y -values.
Find the slope m of the line of best fit using the formula.
image

4. Compute the y -intercept of the line by using the formula:
image

5. Use the slope m and the y -intercept to form the equation of the line. 6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

---
Program:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In Colab: upload the CSV first (folder icon on the left), then run this
df = pd.read_csv('exp_2_dataset_student_scores.csv')
x = df['Hours'].values
y = df['Scores'].values

plt.scatter(x, y)
plt.xlabel('Hours'); plt.ylabel('Scores'); plt.title('Hours vs Scores')
plt.show()

# Building the model (least squares)
x_mean = np.mean(x)
y_mean = np.mean(y)
num = 0
den = 0
for i in range(len(x)):
    num += (x[i]-x_mean)*(y[i]-y_mean)
    den += (x[i]-x_mean)**2
m = num/den
c = y_mean - m*x_mean
print("Slope m =", m, " Intercept c =", c)

# Predictions
y_pred = m*x + c

# Evaluation
r2 = 1 - np.sum((y-y_pred)**2)/np.sum((y-y_mean)**2)
rmse = np.sqrt(np.mean((y-y_pred)**2))
print("R2 =", r2, " RMSE =", rmse)

# Plot
plt.scatter(x, y)
plt.plot([min(x), max(x)], [m*min(x)+c, m*max(x)+c], color='red')
plt.xlabel('Hours'); plt.ylabel('Scores'); plt.title('Regression line')
plt.show()

# Predict for a new value
print("Predicted score for 9.25 hrs:", m*9.25 + c)
```
---
Output:

best fit line


<img width="492" height="813" alt="image" src="https://github.com/user-attachments/assets/444a5d9e-50c5-4ee4-b904-3e9165031a58" />

---
Result:

Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
