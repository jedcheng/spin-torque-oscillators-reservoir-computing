import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt


mackey_glass0 = np.load('mackey_glass_t17.npy')
mackey_glass = mackey_glass0[0:2500]

train_target = mackey_glass[1:2000]
train_input = mackey_glass[0:1999]

regressor = linear_model.LinearRegression()
regressor.fit(train_input.reshape(-1, 1), train_target.reshape(-1, 1))

test_target = mackey_glass[2001:2500]
test_input = mackey_glass[2000:2499]

predictions = regressor.predict(test_input.reshape(-1, 1))

plt.plot(test_target, label='Target')
plt.plot(predictions, label='Predictions')
plt.legend(loc='upper right')
plt.xlabel('Time step')
plt.ylabel('Mackey-Glass value')
plt.title('Mackey-Glass Time Series Prediction with Linear Regression')
plt.savefig('Results/naive_linear_regression.png', dpi=300)
plt.close()


plt.plot(test_target, label='Target')
plt.plot(predictions, label='Predictions')
plt.legend(loc='upper right')
plt.xlabel('Time step')
plt.ylabel('Mackey-Glass value')
plt.title('Mackey-Glass Time Series Prediction with Linear Regression')
plt.xlim(0, 100)
plt.savefig('Results/naive_linear_regression_zoom.png', dpi=300)
plt.close()