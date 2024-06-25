# Data Generating Process
# Lotka-Volterra equation

x0 = 4.
y0 = 2.
Nt = 1000
tmax = 30.

time_space = np.arange(0, tmax, 0.01)

X0 = [x0, y0]

alpha = np.random.uniform(0.1, 1.0)  # Prey birth rate
beta = np.random.uniform(0.01, 0.1)  # Predation rate
gamma = np.random.uniform(0.1, 1.0)  # Predator death rate
delta = np.random.uniform(0.01, 0.1)  # Predator reproduction rate

simulate_process(suffix='train', n_iterations = 800) # 800 * 30 = 24000
simulate_process(suffix='val', n_iterations = 200) # 200 * 30 = 6000
simulate_process(suffix='test', n_iterations = 1) # 1 * 30 = 30