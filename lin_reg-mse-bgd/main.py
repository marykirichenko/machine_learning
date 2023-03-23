import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.linear_model import LinearRegression
from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 andtheta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# calculate closed-form solution
theta_best = [0, 0]
# finding the optimal value of theta using the Normal Equation
# 1.append the column of ones in X to add the bias term
X = np.c_[np.ones(x_train.shape), x_train.reshape((len(x_train), 1))]
Y = y_train.reshape((len(y_train), 1))
theta_best = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
print("Closed-form solution theta: ", theta_best)

# calculate error
MSE = 0
for i in range(len(y_test)):
    MSE += (y_test[i] - (theta_best[0] + theta_best[1]*(x_test[i])))**2

MSE /= len(y_test)
print('MSE: ', MSE)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.title('Simple Linear Regression using closed-form solution')
plt.show()

# standardization
# use Z-score normalization
# calculate standard deviation
sigmaTrain = statistics.stdev(x_train)
meanTrain = statistics.mean(x_train)
y_trains = (y_train - np.average(y_train)) / (np.std(y_train))
x_trains = (x_train - np.average(x_train)) / (np.std(x_train))

y_tests = (y_test - np.average(y_test)) / (np.std(y_test))
x_tests = (x_test - np.average(x_train)) / (np.std(x_train))

#TODO: calculate theta using Batch Gradient Descent

X = np.c_[np.ones(len(x_trains)), x_trains]
x_test = np.c_[np.ones(len(x_tests)), x_tests]

theta_rand = np.random.rand(2,1)

y_c = np.c_[y_trains]
learning_rate = 0.1
for i in range(100):
    g_MSE = ((2/len(x_trains))*X.T).dot(X.dot(theta_rand) - y_c)
    theta_rand = theta_rand - learning_rate * g_MSE

# TODO: calculate error
MSE_sum = 0
for i in range(len(x_tests)):
    MSE_sum += (theta_rand.T.dot(np.c_[np.ones(len(x_tests)), x_tests][i]) - y_tests[i]) ** 2
MSE = 1/len(x_tests) * MSE_sum
print("MSE: ", MSE)


x = np.linspace(min(x_tests), max(x_tests), 100)
y = float(theta_rand[0]) + float(theta_rand[1]) * x
plt.plot(x, y)
plt.scatter(x_tests, y_tests)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()