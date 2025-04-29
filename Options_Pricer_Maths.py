# ===== IMPORTS =====
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from modules.black_scholes import black_scholes
from modules.Binomial import binomial_tree_price
from modules.Binomial import binomial_tree_greeks
from modules.Monte Carlo Simple import monte_carlo_option_pricing
from modules.monte_carlo_longstaff import monte_carlo_longstaff_schwartz

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



