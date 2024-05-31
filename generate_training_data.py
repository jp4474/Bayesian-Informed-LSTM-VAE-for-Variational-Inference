import numpy as np
import pandas as pd
import pickle
import os
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler

def Total_FoB(time, M0 = np.exp(16.7), nu = 0.004, b0 = 20):
    return M0 * (1 + np.exp(-nu * (time - b0) ** 2))

def CAR_negative_MZB(time, M0 = np.exp(14.06), nu = 0.0033, b0 = 20.58):
    return M0 * (1 + np.exp(-nu * (time - b0) ** 2))
  

def ode_function(y, time, alpha, beta, mu, delta, lamb, nu, t0):
    """
    Calculates the derivatives of the system of ordinary differential equations (ODEs)
    for the CAR positive GCB cells in WT and CAR positive MZB cells in WT.

    Returns:
    - list: The derivatives of the system of ODEs, represented as a list of two elements.
    """
    alpha_tau = alpha/(1 + np.exp(nu * (time-t0) ** 2))
    mu_tau = mu/(1 + np.exp(nu * (time-t0) ** 2))
    beta_tau = beta/(1 + np.exp(nu * (time-t0) ** 2))

    # CAR positive GCB cells in WT
    dydt_0 = alpha_tau * Total_FoB(time) - delta * y[0]
    # CAR positive MZB cells in WT
    dydt_1 = mu_tau * Total_FoB(time) + beta_tau * CAR_negative_MZB(time) - lamb * y[1]
    return [dydt_0, dydt_1]

def simulate_process(t0 = 0, tFin = 30, tStep = 0.01, n_iterations = 1000, sequence_length = 30, suffix = 'train'):
    """
    Simulates a process using the given parameters and returns a list of dictionaries containing the simulation results.

    Args:
        t0 (float, optional): The initial time of the simulation. Defaults to 4.
        tFin (float, optional): The final time of the simulation. Defaults to 100.
        tStep (float, optional): The time step for the simulation. Defaults to 0.01.
        n_iterations (int, optional): The number of iterations to run the simulation. Defaults to 1000.

    Returns:
        List[Dict[str, Union[float, List[List[float]]]]]: A list of dictionaries containing the simulation results. Each dictionary has the following keys:
            - 'parameters' (Dict[str, float]): A dictionary containing the simulated parameters.
            - 'data' (List[List[float]]): A list of lists containing the simulation observations. Each inner list contains the time, observation 1, and observation 2.
    """
    if not os.path.exists('data'):
        os.makedirs('data')

    if not os.path.exists(f'data/{suffix}'):
        os.makedirs(f'data/{suffix}')

    if not os.path.exists(f'data/{suffix}/processed'):
        os.makedirs(f'data/{suffix}/processed')

    time_space = np.arange(t0, tFin+tStep, tStep)
    indexes = (time_space % 0.5 == 0)
    result = [0] * n_iterations
    for i in range(n_iterations):
        alpha = np.random.normal(0.01, 0.5)
        beta = np.random.normal(0.01, 0.5)
        mu = np.random.normal(0.01, 0.5)
        nu = np.random.normal(0.01, 0.5)
        delta = np.random.normal(0.8, 0.3)
        lambda_wt = np.random.normal(0.1, 0.3)
        y0 = [24336.98/1e8, 17701.872/1e8]
        sol = odeint(ode_function, y0, time_space, args=(alpha, beta, mu, delta, lambda_wt, nu, t0))
        parameters = {'alpha' : alpha, 'beta' : beta, 'mu' : mu, 'nu' : nu, 'delta' : delta, 'lambda' : lambda_wt}
  
        sol = sol[indexes]
        observations = [[t, obs[0], obs[1]] for t, obs in zip(time_space[indexes], sol)]
        result[i] = {'parameters' : parameters, 'data' : observations}

    with open(f'data/{suffix}/simulated_data.pkl', 'wb') as f:
        pickle.dump(result, f)

    def preprocess_data(suffix = 'train', sequence_length = 30):
        with open(f'data/{suffix}/simulated_data.pkl', 'rb') as f:
            processes = pickle.load(f)
        
        for i in range(len(processes)):
            parameters = processes[i]['parameters']
            data = processes[i]['data']
            #fixme
            for idx in range(len(data) - sequence_length):
                if (idx+sequence_length) > len(data):
                    indexes = list(range(idx, len(data)))
                else:
                    indexes = list(range(idx, idx + sequence_length))
                x = [data[i] for i in indexes]

                with open(f'data/{suffix}/processed/simulated_data_{i}_{idx}.pkl', 'wb') as f:
                    pickle.dump({'parameters' : parameters, 'x' : x}, f)

    preprocess_data(suffix=suffix, sequence_length=sequence_length)

if __name__ == "__main__":    
    simulate_process(t0 = 0, tFin = 30, tStep = 0.01, suffix='train', n_iterations = 1000, sequence_length = 30)
    simulate_process(t0 = 0, tFin = 30, tStep = 0.01, suffix='val', n_iterations = 200, sequence_length = 30)

