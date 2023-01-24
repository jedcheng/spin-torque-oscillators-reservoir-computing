# Reservoir Computing for Time Series Prediction Using Spin Torque Oscillators

A proof of concept reservoir computing using a pool of Spin Torque Oscillators (STOs) to predict time series data (Mackey Glass Equation). The STOs are simulated using the Mumax3. 

## Usage
1. Install [Mumax3](https://mumax.github.io/install.html)
2. Clone this repository
3. Generate the .mx3 files using the `scripts_generator.py` script. 
4. Run the simulation (I used a cluster controlled by `scripts/reservoir.pbs`)

## Customize the parameters
Change the number of STOs and the current applied to each STO in the `scripts_generator.py` file.
Change the simulation parameters in the `template.txt` file.

## Post-processing
1. Run `main.py` to generate the plots and the error metrics.
2. Run `plot_final_result.py` to generate a plot of the MSE for configurations with different number of STOs.

## Results
Predicting the Mackey Glass Equation using 25 STOs. The red line is the prediction, the blue line is the actual data.
![alt text](https://github.com/jedcheng/spin-torque-oscillators-reservoir-computing/blob/main/Results/25_Testing.png?raw=true)


Of course it is vital to compare the results without the reservoir computing but just linear regression. Simple ML models usually "predicts" the data by reusing the previous data when dealing with time series data. 


The prediction plot might looks good.
![alt text](https://github.com/jedcheng/spin-torque-oscillators-reservoir-computing/blob/main/Results/naive_linear_regression.png?raw=true)

But when we zoom in, it is clear that the prediction is just reusing the previous data rendering the model useless.
![alt text](https://github.com/jedcheng/spin-torque-oscillators-reservoir-computing/blob/main/Results/naive_linear_regression_zoom.png?raw=true)



