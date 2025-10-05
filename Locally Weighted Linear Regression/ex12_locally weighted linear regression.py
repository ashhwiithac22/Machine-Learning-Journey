'''
focuses on nearby data points
used in time series forecasting
provides better prediction
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r"D:\ML Lab\tips.csv")

X = data['total_bill'].values.reshape(-1, 1) #selects the total bill as input feature,-1,1 makes it as column vector , many rows one columns
y = data['tip'].values.reshape(-1, 1) #tip is target column
X_mat = np.c_[np.ones(len(X)), X] #includes the intercept term in regression

def lwlr(x, X, y, tau=2.0): #X is data point , X is training data , y is target , tau is the weight
    #weight is high if point is closer to x and weight is low if it is far
    m = X.shape[0] #represents number of row
    W = np.eye(m) #creates an identity matrix of m*m,stores the weight of each training point

    for i in range(m):
        W[i, i] = np.exp(- (x - X[i, 1])**2 / (2 * tau**2)) #gaussian kernal - exponential function
        #x-xi is the distance
    theta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y) #computes the regression coeffecients
    #linalg is the linear algebra,@is thr multiplication in python
    return theta[0] + theta[1]*x
X_test = np.linspace(X.min(), X.max(), 200) #creates 200 points from x_min to x_max to plot smooth curve
y_pred = [lwlr(x, X_mat, y, tau=2.0) for x in X_test]
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_test, y_pred, color='red', label='LWLR fit')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.title('Locally Weighted Linear Regression')
plt.legend()
plt.show()
