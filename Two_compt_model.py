import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import seaborn as sns
from cycler import cycler

def ode_system(X, t, alpha, beta, delta):
    x, y = X
    dotx = alpha * (2 * y + x) - (beta + delta) * x 
    doty = beta * x - (alpha + delta) * y
    return [dotx, doty]

def simulate_ode(parms, x0, y0, t):
    alpha, beta, delta = parms
    return odeint(ode_system, [x0, y0], t, args=(alpha, beta, delta))

if __name__ == "__main__":
    # Parameters
    alpha = 0.0024
    beta  = 1/20
    delta = 0.0005
    x0 = 4500.0
    y0 = 900.0
    eps1 = 0.43
    eps2 = 0.56
    
    Time_pred = np.linspace(0, 300, 300)
    # sample time points from Time_pred for observed data
    Time_obs = np.random.choice(Time_pred, 50, replace=False)

    # Simulate ODE
    X = simulate_ode([alpha, beta, delta], x0, y0, Time_pred)
    df = pd.DataFrame(X, columns=['Comp1', 'Comp2'])
    df['Time'] = Time_pred

    # Add noise
    np.random.seed(7798)
    df_noise = df.copy()
    df_noise['Log_Comp1'] = np.log(df['Comp1'])
    df_noise['Log_Comp2'] = np.log(df['Comp2']) 
    df_noise['Log_Comp1'] += np.random.normal(0, eps1, len(df))
    df_noise['Log_Comp2'] += np.random.normal(0, eps2, len(df))
    df_noise['Comp1_noise'] = np.exp(df_noise['Log_Comp1'])
    df_noise['Comp2_noise'] = np.exp(df_noise['Log_Comp2'])

    # sample data for observed time points
    df_obs = df_noise.loc[df_noise['Time'].isin(Time_obs)]
    
    # Plot
    fig, ax = plt.subplots()

    # Define a color cycle
    colors = plt.cm.tab10.colors
    ax.set_prop_cycle(cycler('color', colors))

    for i, color in zip(range(2), colors):
        sns.scatterplot(data=df_obs, x='Time', y=f'Comp{i+1}_noise', ax=ax, color=color)
        sns.lineplot(data=df, x='Time', y=f'Comp{i+1}', ax=ax, color=color)
        ax.plot([], [], '-o', color=color, label=f'Comp{i+1}')

    plt.title('Two-compartment model')
    plt.xlabel('Time')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.grid(alpha=0.5, linewidth=0.5)
    plt.legend()
    # Save the plot
    plt.savefig('Two_compartment_model.png')

    # Save data
    # with open('data.pkl', 'wb') as f:
    #     pickle.dump(df, f)
    # print(df_obs.head())
    np.save('data.npy', df_obs.to_numpy())
    df_obs.to_csv('data.csv', index=False)