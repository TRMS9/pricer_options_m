############ Fonction pour Monte Carlo - options européennes ##################

def monte_carlo_option_pricing(S, K, T, r, sigma, num_simulations, option_type="call", variance_reduction=False):
    np.random.seed(42)
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if variance_reduction:
        Z_antithetic = -Z
        ST_antithetic = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_antithetic)
        ST = np.concatenate((ST, ST_antithetic))

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    discounted_payoffs = np.exp(-r * T) * payoffs
    option_price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
    lower_bound = option_price - 1.96 * std_error
    upper_bound = option_price + 1.96 * std_error

    return option_price, lower_bound, upper_bound, discounted_payoffs