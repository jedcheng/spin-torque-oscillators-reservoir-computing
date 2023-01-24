# Reservoir Computing for Time Series Prediction Using Spin Torque Oscillators

A proof of concept reservoir computing using a pool of Spin Torque Oscillators (STOs) to predict time series data. The STOs are simulated using the Mumax3. 

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