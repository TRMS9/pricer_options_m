######### Fonction pour Monte Carlo Longstaff-Schwartz ###############
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


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
