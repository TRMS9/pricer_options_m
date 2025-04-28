# ===== IMPORTS =====
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ===== FONCTIONS (au début) =====

################ Fonction pour Black Scholes ##################
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    else:
        raise ValueError("Le type d'option doit être 'call' ou 'put'")

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) *
             (norm.cdf(d2) if option_type == "call" else norm.cdf(-d2))) / 365

    return price, delta, gamma, theta, rho, vega


############## Fonction pour Binomial Tree ##################
def binomial_tree_price(S, K, T, r, sigma, N, option_type="call", american=False):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    stock_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)

    option_tree = np.zeros((N + 1, N + 1))
    if option_type == "call":
        option_tree[:, N] = np.maximum(stock_tree[:, N] - K, 0)
    else:
        option_tree[:, N] = np.maximum(K - stock_tree[:, N], 0)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            hold = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            if american:
                if option_type == "call":
                    exercise = stock_tree[j, i] - K
                else:
                    exercise = K - stock_tree[j, i]
                option_tree[j, i] = max(hold, exercise)
            else:
                option_tree[j, i] = hold

    return option_tree[0, 0]

def binomial_tree_greeks(S, K, T, r, sigma, N, option_type="call", american=False):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    stock_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)

    option_tree = np.zeros((N + 1, N + 1))
    if option_type == "call":
        option_tree[:, N] = np.maximum(stock_tree[:, N] - K, 0)
    else:
        option_tree[:, N] = np.maximum(K - stock_tree[:, N], 0)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            hold = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            if american:
                if option_type == "call":
                    exercise = stock_tree[j, i] - K
                else:
                    exercise = K - stock_tree[j, i]
                option_tree[j, i] = max(hold, exercise)
            else:
                option_tree[j, i] = hold

    price = option_tree[0, 0]

    # Greeks manuellement
    delta = (option_tree[0,1] - option_tree[1,1]) / (stock_tree[0,1] - stock_tree[1,1])

    delta_up = (option_tree[0,2] - option_tree[1,2]) / (stock_tree[0,2] - stock_tree[1,2])
    delta_down = (option_tree[1,2] - option_tree[2,2]) / (stock_tree[1,2] - stock_tree[2,2])
    gamma = (delta_up - delta_down) / ((stock_tree[0,2] - stock_tree[2,2]) / 2)

    theta = (option_tree[1,2] - option_tree[0,0]) / (2 * dt)

    # Vega
    eps_sigma = 0.01
    price_up = binomial_tree_price(S, K, T, r, sigma + eps_sigma, N, option_type, american)
    price_down = binomial_tree_price(S, K, T, r, sigma - eps_sigma, N, option_type, american)
    vega = (price_up - price_down) / (2 * eps_sigma * 100)

    # Rho
    eps_r = 0.0001
    price_up_r = binomial_tree_price(S, K, T, r + eps_r, sigma, N, option_type, american)
    price_down_r = binomial_tree_price(S, K, T, r - eps_r, sigma, N, option_type, american)
    rho = (price_up_r - price_down_r) / (2 * eps_r * 100)

    return price, delta, gamma, theta, vega, rho

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

######### Fonction pour Monte Carlo Longstaff-Schwartz ###############

def monte_carlo_longstaff_schwartz(S, K, T, r, sigma, M, N, option_type="call"):
    np.random.seed(42)
    dt = T / N
    discount = np.exp(-r * dt)

    S_paths = np.zeros((M, N + 1))
    S_paths[:, 0] = S
    for t in range(1, N + 1):
        Z = np.random.standard_normal(M)
        S_paths[:, t] = S_paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    if option_type == "call":
        payoffs = np.maximum(S_paths - K, 0)
    else:
        payoffs = np.maximum(K - S_paths, 0)

    V = payoffs[:, -1]

    for t in range(N-1, 0, -1):
        itm = payoffs[:, t] > 0
        X = S_paths[itm, t]
        Y = V[itm] * discount

        if len(X) == 0:
            continue

        A = np.vstack([np.ones(len(X)), X, X**2]).T
        coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
        continuation_value = coeffs[0] + coeffs[1]*X + coeffs[2]*X**2

        exercise = payoffs[itm, t]
        V[itm] = np.where(exercise > continuation_value, exercise, V[itm] * discount)

    price = np.mean(V) * np.exp(-r * dt)

    return price, S_paths, V

# === STREAMLIT APP ===

st.set_page_config(page_title="Pricing Options App", layout="wide")
st.title("📈 Application de Pricing d'Options")

# Création des onglets
tab1, tab2, tab3, tab4 = st.tabs(["Black-Scholes", "Binomial Tree", "Monte Carlo Européen", "Monte Carlo Longstaff-Schwartz"])

# === Onglet 1 : Black-Scholes ===
with tab1:
    st.header("📈 Pricing avec Black-Scholes")

    # Inputs Streamlit
    S = st.number_input("Prix du sous-jacent (S)", value=100.0)
    K = st.number_input("Prix d'exercice (K)", value=100.0)
    T = st.number_input("Temps jusqu'à maturité (T, en années)", value=1.0)
    r = st.number_input("Taux sans risque (r)", value=0.05)
    sigma = st.number_input("Volatilité (σ)", value=0.2)
    option_type = st.selectbox("Type d'option", ("call", "put"))

    if st.button("Calculer avec Black-Scholes"):

        # Calcul principal
        price, delta, gamma, theta, rho, vega = black_scholes(S, K, T, r, sigma, option_type)

        # Affichage des résultats
        st.success(f"💰 Prix de l'option : {price:.4f} €")
        st.write(f"Delta : {delta:.4f} | Gamma : {gamma:.6f} | Theta : {theta:.4f}")
        st.write(f"Rho : {rho:.4f} | Vega : {vega:.4f}")

        # Courbes pour différents prix
        S_values = np.linspace(S * 0.5, S * 1.5, 100)
        results = [black_scholes(Si, K, T, r, sigma, option_type) for Si in S_values]
        prices, deltas, gammas, thetas, rhos, vegas = zip(*results)

        # Création des graphiques
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        axs[0, 0].plot(S_values, prices)
        axs[0, 0].set_title("Prix de l'option")

        axs[0, 1].plot(S_values, deltas, color='r')
        axs[0, 1].set_title("Delta")

        axs[0, 2].plot(S_values, gammas, color='g')
        axs[0, 2].set_title("Gamma")

        axs[1, 0].plot(S_values, thetas, color='m')
        axs[1, 0].set_title("Theta")

        axs[1, 1].plot(S_values, rhos, color='c')
        axs[1, 1].set_title("Rho")

        axs[1, 2].plot(S_values, vegas, color='y')
        axs[1, 2].set_title("Vega")

        for ax in axs.flat:
            ax.grid()
            ax.set_xlabel("Prix du sous-jacent (S)")
            ax.legend(["Valeur"])
        
        plt.tight_layout()
        st.pyplot(fig)

# === Onglet 2 : Binomial Tree ===
with tab2:
    st.header("🌳 Pricing avec Binomial Tree")

    # Inputs
    S = st.number_input("Prix du sous-jacent (S)", value=100.0, key="bin_s")
    K = st.number_input("Prix d'exercice (K)", value=100.0, key="bin_k")
    T = st.number_input("Temps jusqu'à maturité (T, en années)", value=1.0, key="bin_t")
    r = st.number_input("Taux sans risque (r)", value=0.05, key="bin_r")
    sigma = st.number_input("Volatilité (σ)", value=0.2, key="bin_sigma")
    N = st.number_input("Nombre de pas (N)", min_value=10, value=50, step=1, key="bin_N")
    option_type = st.selectbox("Type d'option", ("call", "put"), key="bin_type")
    american = st.checkbox("Option américaine ?", value=False)

    if st.button("Calculer avec Binomial Tree"):

        # Calcul principal
        price, delta, gamma, theta, vega, rho = binomial_tree_greeks(S, K, T, r, sigma, int(N), option_type, american)

        st.success(f"💰 Prix de l'option {option_type.capitalize()} : {price:.4f} €")
        st.write(f"Delta : {delta:.4f} | Gamma : {gamma:.6f}")
        st.write(f"Theta : {theta:.4f} | Vega : {vega:.4f} | Rho : {rho:.4f}")

        # Courbes pour différents prix S
        S_values = np.linspace(S * 0.5, S * 1.5, 100)
        results = [binomial_tree_greeks(Si, K, T, r, sigma, int(N), option_type, american) for Si in S_values]
        prices, deltas, gammas, thetas, vegas, rhos = zip(*results)

        # Graphiques
        fig, axs = plt.subplots(3, 2, figsize=(16, 15))

        axs[0, 0].plot(S_values, prices, label="Prix")
        axs[0, 0].set_title("Prix de l'option")

        axs[0, 1].plot(S_values, deltas, color='r', label="Delta")
        axs[0, 1].set_title("Delta")

        axs[1, 0].plot(S_values, gammas, color='g', label="Gamma")
        axs[1, 0].set_title("Gamma")

        axs[1, 1].plot(S_values, thetas, color='m', label="Theta")
        axs[1, 1].set_title("Theta")

        axs[2, 0].plot(S_values, vegas, color='y', label="Vega")
        axs[2, 0].set_title("Vega")

        axs[2, 1].plot(S_values, rhos, color='c', label="Rho")
        axs[2, 1].set_title("Rho")

        for ax in axs.flat:
            ax.grid()
            ax.legend()
            ax.set_xlabel("Prix du sous-jacent (S)")

        plt.tight_layout()
        st.pyplot(fig)
# === Onglet 3 : Monte Carlo Européen ===
with tab3:
    st.header("🎲 Pricing avec Monte Carlo Options Européennes")

    # Inputs Streamlit
    S = st.number_input("Prix du sous-jacent (S)", value=100.0, key="mc_s")
    K = st.number_input("Prix d'exercice (K)", value=100.0, key="mc_k")
    T = st.number_input("Temps jusqu'à maturité (T, en années)", value=1.0, key="mc_t")
    r = st.number_input("Taux sans risque (r)", value=0.05, key="mc_r")
    sigma = st.number_input("Volatilité (σ)", value=0.2, key="mc_sigma")
    num_simulations = st.number_input("Nombre de simulations Monte Carlo", min_value=100, value=10000, step=100, key="mc_num")
    option_type = st.selectbox("Type d'option", ("call", "put"), key="mc_type")
    variance_reduction = st.checkbox("Utiliser la réduction de variance (antithétique) ?", value=False)

    if st.button("Calculer avec Monte Carlo Européen"):

        # Calcul
        option_price, lower_bound, upper_bound, discounted_payoffs = monte_carlo_option_pricing(S, K, T, r, sigma, int(num_simulations), option_type, variance_reduction)

        # Affichage des résultats
        st.success(f"💰 Prix estimé de l'option {option_type.capitalize()} : {option_price:.4f} €")
        st.write(f"Intervalle de confiance 95% : [{lower_bound:.4f} €, {upper_bound:.4f} €]")

        # Graphique des payoffs
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(discounted_payoffs, bins=100, density=True, alpha=0.6, color='skyblue', label='Distribution des payoffs actualisés')
        ax.axvline(x=lower_bound, color='red', linestyle='--', label='Borne inférieure 95%')
        ax.axvline(x=upper_bound, color='red', linestyle='--', label='Borne supérieure 95%')
        ax.axvline(x=option_price, color='black', linestyle='-', label='Prix moyen')

        ax.set_title(f"Distribution des payoffs actualisés - {option_type.capitalize()} Monte Carlo")
        ax.set_xlabel("Payoff actualisé")
        ax.set_ylabel("Densité")
        ax.legend()
        ax.grid()

        st.pyplot(fig)

# === Onglet 4 : Monte Carlo Longstaff-Schwartz ===
with tab4:
    st.header("🎲📚 Pricing avec Monte Carlo Longstaff-Schwartz")

    # Inputs Streamlit
    S = st.number_input("Prix du sous-jacent (S)", value=100.0, key="ls_s")
    K = st.number_input("Prix d'exercice (K)", value=100.0, key="ls_k")
    T = st.number_input("Temps jusqu'à maturité (T, en années)", value=1.0, key="ls_t")
    r = st.number_input("Taux sans risque (r)", value=0.05, key="ls_r")
    sigma = st.number_input("Volatilité (σ)", value=0.2, key="ls_sigma")
    M = st.number_input("Nombre de simulations Monte Carlo", min_value=100, value=10000, step=100, key="ls_m")
    N = st.number_input("Nombre de dates d'exercice (pas de temps)", min_value=2, value=50, step=1, key="ls_n")
    option_type = st.selectbox("Type d'option", ("call", "put"), key="ls_type")

    if st.button("Calculer avec Monte Carlo Longstaff-Schwartz"):

        # Calcul
        price, S_paths, V = monte_carlo_longstaff_schwartz(S, K, T, r, sigma, int(M), int(N), option_type)

        # Affichage du résultat
        st.success(f"💰 Prix estimé de l'option américaine {option_type.capitalize()} : {price:.4f} €")

        # 1. Graphique des trajectoires simulées
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        time_grid = np.linspace(0, T, N+1)
        for i in range(min(20, S_paths.shape[0])):  # Affiche max 20 trajectoires
            ax1.plot(time_grid, S_paths[i], lw=1)
        ax1.set_title("Exemples de trajectoires simulées du sous-jacent")
        ax1.set_xlabel("Temps (années)")
        ax1.set_ylabel("Prix du sous-jacent")
        ax1.grid()
        st.pyplot(fig1)

        # 2. Graphique de la distribution des valeurs actualisées finales
        discounted_values = V * np.exp(-r * T/N)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.hist(discounted_values, bins=50, color='skyblue', edgecolor='black', density=True)
        ax2.set_title("Distribution des valeurs actualisées des payoffs")
        ax2.set_xlabel("Valeur actualisée du payoff")
        ax2.set_ylabel("Densité")
        ax2.grid()
        st.pyplot(fig2)



