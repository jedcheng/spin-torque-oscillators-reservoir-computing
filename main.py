import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from os.path import split, splitext, join

from scipy import interpolate
from scipy.fft import fftfreq, fft

from sklearn import linear_model
from scipy import signal
from sklearn import linear_model

from multiprocessing import Pool
from itertools import repeat

def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx

def average_each_interval(mag, step_every5ns):
    mag_ = []
    
    for i in range(len(mag)):
        mag_i_ = mag[i].reshape(-1, step_every5ns).mean(axis=1)
        
        mag_.append(mag_i_)
        
    return np.array(mag_)

def preprocess(folder_path, steps_every5ns=5):
    no_of_stos = int(split(splitext(folder_path)[0])[1])**2
    
    df = pd.read_csv(join(folder_path, 'table.txt'),sep='\t')
    t0 = df['# t (s)'].to_numpy()
    magnetizations = []
    for i in range(1, no_of_stos+1):
        magnetizations.append(df['m.region{}y ()'.format(i)].to_numpy())

    t_new = np.arange(700e-9, 1.32e-5, 5e-9/steps_every5ns)

    mag_new = []

    for i in range(no_of_stos):
        f = interpolate.interp1d(t0, magnetizations[i])
        mag_new_ = f(t_new)
        mag_new.append(mag_new_)
        plt.plot(t_new, mag_new_, label='STO {}'.format(i+1))
        
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetization')
    plt.title(f'Magnetization of {no_of_stos} STOs')
    plt.savefig(f'Results/{no_of_stos}_Magnetization.png', dpi=300)
    plt.close()
    return mag_new


def main(folder_path, step_every5ns):
    no_of_stos = int(split(splitext(folder_path)[0])[1])**2
    
    mackey_glass0 = np.load('mackey_glass_t17.npy')
    mackey_glass = mackey_glass0[0:2500]
    

    mag0 = preprocess(folder_path, step_every5ns)


    # Hilbert transform to get the amplitude
    amp = []

    for i in range(0, len(mag0)):
        amp.append(np.abs(signal.hilbert(mag0[i])))

    amp = np.array(amp)
    amp1 = average_each_interval(amp, step_every5ns)


    # Training split
    target = mackey_glass[0:2000]
    target = target[1:] 

    input = amp1[:, 0:len(target)]

    # Linear Regression
    regressor = linear_model.LinearRegression()
    regressor.fit(input.T, target)
    regressor.score(input.T, target)

    # Visualize the training
    predictions = regressor.predict(input.T)
    mse_training = np.mean((target - predictions)**2)
    
    plt.plot(target, label='Target')
    plt.plot(predictions*2, label='Predictions')
    plt.legend(loc='upper right')

    plt.title(f'Mackey-Glass Prediction in Training using {no_of_stos} STOs')
    plt.xlabel('Time Steps')
    plt.ylabel('Mackey-Glass value')
    plt.savefig(f'Results/{no_of_stos}_Training.png', dpi=300)
    plt.close()

    

    # Testing split
    testing_input = amp1[:, 2000:2500-1]
    testing_target = mackey_glass[2001:2500]
    
    # Predictions
    testing_prediction_ = regressor.predict(testing_input.T)
    testing_predictions = np.array([0])
    testing_predictions = np.append(testing_predictions, testing_prediction_)

    mse_testing = np.mean((testing_target - testing_predictions[1:])**2)
    
    # Visualize the testing
    plt.plot(testing_target, label='Target')
    plt.plot(testing_predictions, label='Predictions')
    plt.legend(loc='upper right')
    plt.title(f'Mackey-Glass Prediction using {no_of_stos} STOs')
    plt.ylabel('Mackey-Glass value')
    plt.xlabel('Time step')
    plt.savefig(f'Results/{no_of_stos}_Testing.png', dpi=300)
    plt.close()

    # Save the results
    with open(f'Results/{no_of_stos}_results.txt', 'w') as f:
        f.write('MSE Training: {}\n'.format(mse_training))
        f.write('MSE Testing: {}\n'.format(mse_testing))
        
    print('Done with {} STOs'.format(no_of_stos))
        
        
if __name__ == '__main__':
    files = ['scripts/{}.out'.format(i) for i in range(2, 6)]
    steps = 5
    with Pool() as p:
        p.starmap(main, zip(files, repeat(steps)))
    