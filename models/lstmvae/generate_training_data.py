import numpy as np
import pandas as pd
import pickle
import os
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# def Total_FoB(time, M0 = np.exp(16.7), nu = 0.004, b0 = 20):
#     return M0 * (1 + np.exp(-nu * (time - b0) ** 2))

# def CAR_negative_MZB(time, M0 = np.exp(14.06), nu = 0.0033, b0 = 20.58):
#     return M0 * (1 + np.exp(-nu * (time - b0) ** 2))

# def ode_function(y, time, alpha, beta, mu, delta, lamb, nu, t0):
#     """
#     Calculates the derivatives of the system of ordinary differential equations (ODEs)
#     for the CAR positive GCB cells in WT and CAR positive MZB cells in WT.

#     Returns:
#     - list: The derivatives of the system of ODEs, represented as a list of two elements.
#     """
#     alpha_tau = alpha/(1 + np.exp(nu * (time-t0) ** 2))
#     mu_tau = mu/(1 + np.exp(nu * (time-t0) ** 2))
#     beta_tau = beta/(1 + np.exp(nu * (time-t0) ** 2))

#     # CAR positive GCB cells in WT
#     dydt_0 = alpha_tau * Total_FoB(time) - delta * y[0]
#     # CAR positive MZB cells in WT
#     dydt_1 = mu_tau * Total_FoB(time) + beta_tau * CAR_negative_MZB(time) - lamb * y[1]
#     return [dydt_0, dydt_1]

# def simulate_process(t0 = 0, tFin = 30, tStep = 0.01, n_iterations = 1000, sequence_length = 30, suffix = 'train', as_pickle = True):
#     """
#     Simulates a process using the given parameters and returns a list of dictionaries containing the simulation results.

#     Args:
#         t0 (float, optional): The initial time of the simulation. Defaults to 4.
#         tFin (float, optional): The final time of the simulation. Defaults to 100.
#         tStep (float, optional): The time step for the simulation. Defaults to 0.01.
#         n_iterations (int, optional): The number of iterations to run the simulation. Defaults to 1000.

#     Returns:
#         List[Dict[str, Union[float, List[List[float]]]]]: A list of dictionaries containing the simulation results. Each dictionary has the following keys:
#             - 'parameters' (Dict[str, float]): A dictionary containing the simulated parameters.
#             - 'data' (List[List[float]]): A list of lists containing the simulation observations. Each inner list contains the time, observation 1, and observation 2.
#     """
#     if not os.path.exists('data'):
#         os.makedirs('data')

#     if not os.path.exists(f'data/{suffix}'):
#         os.makedirs(f'data/{suffix}')

#     if not os.path.exists(f'data/{suffix}/processed'):
#         os.makedirs(f'data/{suffix}/processed')

#     time_space = np.arange(t0, tFin+tStep, tStep)
#     indexes = (time_space % 0.5 == 0)
#     result = [0] * n_iterations

#     df = pd.DataFrame()
#     for i in range(n_iterations):
#         #sol = np.array([-1])
#         #while np.any(sol < 0):
#         alpha, beta, mu, delta, lambda_wt, nu = 0, 0, 0, 0, 0, 0
#         while(np.any(np.array([alpha, beta, mu, delta, lambda_wt, nu]) < 0)):
#             alpha = np.random.normal(0.01, 0.5)
#             beta = np.random.normal(0.01, 0.5)
#             mu = np.random.normal(0.01, 0.5)
#             nu = np.random.normal(0.01, 0.5)
#             delta = np.random.normal(0.8, 0.3)
#             lambda_wt = np.random.normal(0.1, 0.3)

#         y0 = [24336.98, 17701.872]
#         sol = odeint(ode_function, y0, time_space, args=(alpha, beta, mu, delta, lambda_wt, nu, t0))
#         parameters = {'alpha' : alpha, 'beta' : beta, 'mu' : mu, 'nu' : nu, 'delta' : delta, 'lambda' : lambda_wt}
#         alpha = np.random.normal(0.01, 0.5)
#         beta = np.random.normal(0.01, 0.5)
#         mu = np.random.normal(0.01, 0.5)
#         nu = np.random.normal(0.01, 0.5)
#         delta = np.random.normal(0.8, 0.3)
#         lambda_wt = np.random.normal(0.1, 0.3)
#         y0 = [24336.98, 17701.872]
#         sol = odeint(ode_function, y0, time_space, args=(alpha, beta, mu, delta, lambda_wt, nu, t0))
#         parameters = {'alpha' : alpha, 'beta' : beta, 'mu' : mu, 'nu' : nu, 'delta' : delta, 'lambda' : lambda_wt}

#         sol = sol[indexes]
#         observations = [[t, obs[0], obs[1]] for t, obs in zip(time_space[indexes], sol)]

#         result[i] = {'parameters' : parameters, 'data' : observations}

#     with open(f'data/{suffix}/simulated_data.pkl', 'wb') as f:
#         pickle.dump(result, f)

#     def preprocess_data(suffix = 'train', sequence_length = 30):
#         with open(f'data/{suffix}/simulated_data.pkl', 'rb') as f:
#             processes = pickle.load(f)
        
#         for i in range(len(processes)):
#             parameters = processes[i]['parameters']
#             data = processes[i]['data']
#             #fixme
#             for idx in range(len(data) - sequence_length):
#                 if (idx+sequence_length) > len(data):
#                     indexes = list(range(idx, len(data)))
#                 else:
#                     indexes = list(range(idx, idx + sequence_length))
#                 x = [data[i] for i in indexes]

#                 with open(f'data/{suffix}/processed/simulated_data_{i}_{idx}.pkl', 'wb') as f:
#                     pickle.dump({'parameters' : parameters, 'x' : x}, f)

#     if as_pickle:
#         preprocess_data(suffix=suffix, sequence_length=sequence_length)

# def parse_as_dataframe():
#     with open('data/train/simulated_data.pkl', 'rb') as f:
#         processes = pickle.load(f)

#     data_process_1_dict = {}
#     data_process_2_dict = {}
#     for i in range(len(processes)):
#         # parameters = processes[i]['parameters']
#         data = processes[i]['data']
#         data_process_1_dict[f'Process 1_{i}'] = [item[1] for item in data] 
#         data_process_2_dict[f'Process 1_{i}'] = [item[2] for item in data]

#     df_1 = pd.DataFrame(data_process_1_dict)
#     df_1['time'] = np.arange(0, 30.01, 0.5)
#     df_2 = pd.DataFrame(data_process_2_dict)
#     df_2['time'] = np.arange(0, 30.01, 0.5)

#     return df_1, df_2

# def generate_sinusoidal(t0 = 0, tFin = 30, tStep = 0.01, suffix='train', n_iterations = 1000, sequence_length = 30, as_pickle=False):
#     if not os.path.exists('data'):
#         os.makedirs('data')

#     if not os.path.exists(f'data/{suffix}'):
#         os.makedirs(f'data/{suffix}')

#     if not os.path.exists(f'data/{suffix}/processed'):
#         os.makedirs(f'data/{suffix}/processed')

#     time_space = np.arange(t0, tFin+tStep, tStep)
#     #indexes = (time_space % 0.5 == 0)
#     result = [0] * n_iterations

#     for i in range(n_iterations):
#         a = np.random.normal(5, 1.5)
#         b = np.random.normal(1, 0.3)

#         y = a * np.sin(2*np.pi/b * time_space)

#         x = [a,b]

#         # with open(f'data/{suffix}/processed/sinusoidal_{i}_{j}.pkl', 'wb') as f:
#         #     pickle.dump({'x' : x, 'y' : y}, f)

#         for j in range(len(time_space) - sequence_length):
#             if (j+sequence_length) > len(time_space):
#                 y_output = y[j:len(time_space)]
#             else:
#                 y_output = y[j:j+sequence_length]

#             with open(f'data/{suffix}/processed/sinusoidal_{i}_{j}.pkl', 'wb') as f:
#                 pickle.dump({'x' : x, 'y' : y_output}, f)
                                
#         # preprocess(df, suffix=suffix, sequence_length=sequence_length)




def ode_system(X, alpha, beta, delta, gamma):
    # Lotka-Volterra equation
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-gamma + delta * x)
    return np.array([dotx, doty])

def simulate_process(sequence_length = 30, suffix='train', n_iterations = 1000):
    if not os.path.exists('data'):
        os.makedirs('data')

    if not os.path.exists(f'data/{suffix}'):
        os.makedirs(f'data/{suffix}')

    if not os.path.exists(f'data/{suffix}/processed'):
        os.makedirs(f'data/{suffix}/processed')

    x0 = 30.
    y0 = 4.
    tmax = 20.
    time_space = np.arange(0, tmax, 0.01)
    X0 = [x0, y0]

    for i in range(n_iterations):
        alpha = np.random.uniform(0.1, 1.0)  # Prey birth rate
        beta = np.random.uniform(0.01, 0.1)  # Predation rate
        gamma = np.random.uniform(0.1, 1.0)  # Predator death rate
        delta = np.random.uniform(0.01, 0.1)  # Predator reproduction rate

        res = odeint(ode_system, X0, time_space, args = (alpha, beta, delta, gamma))

        # Extract the individual trajectories.
        x, y = res.T

        # Determine the size of the array
        size = len(x)
        # Create an array of zeros
        index = np.zeros(size)
        # Set every 50th element to 1
        index[::10] = 1

        x = x[index.astype(bool)]
        y = y[index.astype(bool)]

        if sequence_length > 0 :
            for j in range(len(x) - sequence_length):
                if (j+sequence_length) > len(x):
                    y_output = [x[j:len(x)], y[j:len(x)]]
                else:
                    y_output = [x[j:j+sequence_length], y[j:j+sequence_length]]

                with open(f'data/{suffix}/processed/simulated_data_{i}_{j}.pkl', 'wb') as f:
                    pickle.dump({'parameters' : {'alpha' : alpha, 'beta' : beta, 'delta' : delta, 'gamma' : gamma}, 
                                'y' : y_output}, f)
        else:
            with open(f'data/{suffix}/processed/simulated_data_{i}.pkl', 'wb') as f:
                pickle.dump({'parameters' : {'alpha' : alpha, 'beta' : beta, 'delta' : delta, 'gamma' : gamma}, 
                            'y' : [x,y]}, f)


if __name__ == "__main__":    
    simulate_process(sequence_length = -1, suffix='train', n_iterations = 8000) # 800 * 30 = 24000
    simulate_process(sequence_length = -1, suffix='val', n_iterations = 2000) # 200 * 30 = 6000
    simulate_process(sequence_length = -1, suffix='test', n_iterations = 10) # 1 * 30 = 30