############## Fonction pour Binomial Tree ##################
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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

    # Greeks
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
