from scripts.useful_functions import G
from scipy import stats
from numpy import sqrt, log
if __name__ == '__main__':

    # option parameters
    T = 1
    S0 = 102.22222
    H = 90  # limit
    K = 100.0  # strike
    r_premia = 10  # annual interest rate

    # Heston model parameters
    V0 = 0.1  # initial volatility
    kappa = 2.0  # heston parameter, mean reversion
    theta = 0.1  # heston parameter, long-run variance
    sigma = omega = 0.2  # heston parameter, volatility of variance.
    # Omega is used in variance tree, sigma - everywhere else
    rho = 0.5  # heston parameter #correlation


def generate_heston_trajectory_return(T, S0, H, K, r_premia, V0, kappa, theta, sigma, rho, N = 500):
    """simulates Heston monte-carlo for Down-and-out put directly through equations"""
    r = log(r_premia / 100 + 1)
    dt = float(T)/float(N)
    sqrt_dt = sqrt(dt)
    # trajectory started

    # initials
    S_t = S0
    V_t = V0
    t = 0

    while t <= T:
        # random walk for V
        random_value_for_V = stats.norm.rvs()
        dZ_V = random_value_for_V * sqrt_dt

        # random walk for S + correlation
        random_value_for_S = stats.norm.rvs()
        random_value_for_S = rho * random_value_for_V + sqrt(1 - pow(rho, 2)) * random_value_for_S
        dZ_S = random_value_for_S * sqrt_dt

        # equation for V
        dV_t = kappa * (theta - V_t) * dt + sigma * sqrt(V_t) * sqrt_dt * dZ_V
        V_t += dV_t
        # equation for S
        dS_t = S_t * r * dt + S_t * sqrt(V_t) * dZ_S
        S_t += dS_t
        # trajectory ended
        t += dt
        # check barrier crossing on each step
        if S_t <= H:
            return 0
    return G(S_t, K)


def calculate_heston_mc_price(T, S0, H, K, r_premia, V0, kappa, theta, sigma, rho, trajectories=5000):
    monte_carlo_price = 0
    for i in range(trajectories):
        monte_carlo_price += generate_heston_trajectory_return(T, S0, H, K, r_premia, V0, kappa, theta, sigma, rho)
    return monte_carlo_price * pow(trajectories, -1)

if __name__ == '__main__':
    print(T, S0, H, K, r_premia, V0, kappa, theta, sigma, rho)
    print(calculate_heston_mc_price(T, S0, H, K, r_premia, V0, kappa, theta, sigma, rho, trajectories=500))
