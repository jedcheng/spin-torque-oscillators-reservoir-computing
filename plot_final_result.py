import numpy as np
from matplotlib import pyplot as plt


training = []
testing = []
no_of_stos = []

for i in range(2, 6):
    with open(f'./Results/{i**2}_results.txt', 'r') as f:
        training_ = float(f.readline().split(' ')[-1])
        testing_ = float(f.readline().split(' ')[-1])
        
        training.append(training_)
        testing.append(testing_)
        
        no_of_stos.append(i**2)
        



plt.plot(no_of_stos, training, label='Training', marker='o')
plt.plot(no_of_stos, testing, label='Testing', marker='o')
plt.xlabel('Number of STOs')
plt.ylabel('MSE')
plt.legend(loc='upper right')
plt.title('MSE for different number of STOs')
plt.savefig('Results/Results.png', dpi=300)