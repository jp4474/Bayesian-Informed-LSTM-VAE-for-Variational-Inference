import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import multiprocessing
import seaborn as sns
import time

import warnings

# Suppress the FutureWarning about DataFrame.swapaxes
warnings.filterwarnings("ignore", category=FutureWarning, message="'DataFrame.swapaxes' is deprecated and will be removed in a future version. Please use 'DataFrame.transpose' instead.")

# The ODE system
def ode_system(X, t, alpha, beta, delta):
    x, y = X
    dotx = alpha * (y + x) - (beta + delta) * x 
    doty = beta * x - (alpha + delta) * y
    return [dotx, doty]

# function to numerically integrate the ODE system
def simulate_ode(parms, x0, y0, t):
    alpha, beta, delta = parms
    return odeint(ode_system, [x0, y0], t, args=(alpha, beta, delta))

# function to saample the parameters from a truncated normal distribution
# low and upp are the lower and upper bounds of the distribution
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

# function to calculate the inverse logit (it is the same as the sigmoid function)
def expit(x):
    return 1 / (1 + np.exp(-x))

# function to sample the parameters and simulate the data
def simulate_and_sample(prior_means, prior_sds, Time_pred):
    # sample the parameters from the prior distribution
    # We are assuming that the parameters are non negative and follow a normal distribution
    alpha = np.random.uniform(prior_means[0], prior_sds[0], 1)[0]
    beta = np.random.uniform(prior_means[1], prior_sds[1], 1)[0]
    delta = np.random.uniform(prior_means[2], prior_sds[2], 1)[0]
    x0_log = np.random.normal(prior_means[3], prior_sds[3])
    y0_log = np.random.normal(prior_means[4], prior_sds[4])
    eps1 = get_truncated_normal(prior_means[5], prior_sds[5], 0, np.inf).rvs()
    eps2 = get_truncated_normal(prior_means[6], prior_sds[6], 0, np.inf).rvs()

    # initial conditions
    x0 = np.exp(x0_log)
    y0 = np.exp(y0_log)

    # Simulate ODE
    X = simulate_ode([alpha, beta, delta], x0, y0, Time_pred)
    df = pd.DataFrame(X, columns=['Comp_x', 'Comp_y'])

    # data mapping
    df['total_counts'] = df['Comp_x'] + df['Comp_y']
    df['frac_x'] = df['Comp_x'] / df['total_counts']
    
    # clean the data -- delete the columns containing the compartment counts
    df = df.drop(columns=['Comp_x', 'Comp_y'])

    # Data transformation
    df['total_counts'] = np.log(df['total_counts'])  # log transform the total counts
    df['frac_x'] = np.log(df['frac_x'] / (1 - df['frac_x'])) # logit transform the fraction of x

    # Add noise
    df['total_counts'] += np.random.normal(0, eps1, len(df))
    df['frac_x'] += np.random.normal(0, eps2, len(df))

    # Inverse transformation
    df['total_counts'] = np.exp(df['total_counts'])
    df['frac_x'] = expit(df['frac_x'])
    
    # Rename the columns
    df['Time'] = Time_pred

    return df

# function to simulate the data using parallel processing
def sim(niter=10000, Tend=300, Tnum=200, num_cores=4):
    # Priors -
    # We assume that the parameters follow a normal distribution with the following mean and sd
    # parameters: alpha, beta, delta, x0, y0, eps1, eps2
    # prior_means = [0.008, 0.01, 0.003, 6.0, 9.0, 0.0, 0.0]
    # prior_sds = [0.01, 0.01, 0.01, 0.8, 1.5, 0.1, 0.15]
    # prior_sds = [0.02, 0.02, 0.01, 0.8, 1.5, 0.1, 0.15]

    prior_a = [0, 0, 0, 6.0, 9.0, 0.0, 0.0]
    prior_b = [0.01, 0.02, 0.01, 0.8, 1.5, 0.1, 0.15]
   
    # time points for prediction
    Time_pred = np.linspace(0, Tend, Tnum).astype(int)

    # Number of parallel simulations
    num_simulations = niter

    # Run simulations in parallel using multiprocessing Pool
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.starmap(simulate_and_sample, [(prior_a, prior_b, Time_pred) for _ in range(num_simulations)])

    # Combine results into a single DataFrame
    df_parallel = pd.concat(results, ignore_index=True)

    # Add a column to identify the simulation number
    df_parallel['Simulation'] = np.repeat(np.arange(num_simulations), len(Time_pred))

    return df_parallel

# function to write the data to a csv file using parallelization
def write_chunk_to_csv(args):
        chunk, filename = args
        chunk.to_csv(filename, mode='a', header=False, index=False)


if __name__ == "__main__":
    # Run the main function and time it
    start_time = time.time()
    
    # define number of cores for parallel processing
    num_cores = multiprocessing.cpu_count() - 2
    print(f"Number of cores: {num_cores}")
    # run the sim function
    sims_parallel = sim(niter=10000, num_cores=num_cores) 
    end_time = time.time()
    print(f"Execution time for Sim function: {end_time - start_time} seconds")

    # import the observed time points
    obs_data = pd.read_csv('data/data.csv')
    Time_obs = obs_data['Time'].values
    #Time_obs = np.loadtxt('datafiles/artf_observed_timepoints.csv', delimiter=',', skiprows=1)
    
    # sample data for observed time points
    sims_obs = sims_parallel[sims_parallel['Time'].isin(Time_obs)]
        
    # Save the data to a csv file for later use in the analysis
    start_time2 = time.time()

    # Split the data into chunks
    chunks = np.array_split(sims_obs, 50)  # split the data into 50 chunks -- adjust as needed

    # Save the data to a csv file for later use using multiprocessing
    output_file = 'artf_noisy_data.csv'
    with open(output_file, 'w') as f:
        sims_obs.head(0).to_csv(f, index=False)  # Write header

    with multiprocessing.Pool(num_cores) as pool:
        pool.map(write_chunk_to_csv, [(chunk, output_file) for chunk in chunks])
    
    end_time2 = time.time()
    print(f"Execution time for write function: {end_time2 - start_time2} seconds")

    # if sim_obs has negative values, then we need to re-run the simulation
    # Throw an error if there are negative values
    if sims_obs[sims_obs < 0].any().any():
        raise ValueError("Negative values in the data. Please re-run the simulation")

    # Plot a random sample of the data

    # filter based on a random simulation number
    # random_simulation = np.random.choice(sims_obs['Simulation'])
    # print(random_simulation)
    sims_obs_sample = sims_obs #[sims_obs['Simulation'] == random_simulation]

    # plot sample data as subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(x='Time', y='total_counts', data=sims_obs_sample, ax=ax[0])
    ax[0].set_title('Total Counts')
    ax[0].set_yscale('log')
    sns.scatterplot(x='Time', y='frac_x', data=sims_obs_sample, ax=ax[1])
    ax[1].set_title('Fraction of x')
    ax[1].set_ylim([0, 1])
    plt.savefig('plots/Two_compartment_model_parallel_sample.png')

    # Plot all the data as suboplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.lineplot(x='Time', y='total_counts', data=sims_obs, ax=ax[0])
    ax[0].set_title('Total Counts')
    ax[0].set_yscale('log')
    sns.lineplot(x='Time', y='frac_x', data=sims_obs, ax=ax[1])
    ax[1].set_title('Fraction of x')
    ax[1].set_ylim([0, 1])
    plt.savefig('plots/Two_compartment_model_parallel_all.png')
    
    print("Done!")