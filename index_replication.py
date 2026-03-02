"""
=============================================================================
  Index Replication Optimizer — Partial Physical Replication
  Auteur  : Mohamed Amine El Aouadi
  Stack   : Python 3, NumPy, Pandas, SciPy, Matplotlib
  Objectif: Répliquer un indice de référence (CAC 40 synthétique) avec un
            sous-ensemble de N titres, en minimisant la tracking error ex-ante
            via optimisation quadratique sous contraintes (SLSQP).
  Structure:
    1. Génération de données synthétiques (modèle factoriel multi-facteurs)
    2. Sélection du panier (greedy par corrélation avec l'indice)
    3. Optimisation des poids (minimisation TE sous contraintes)
    4. Évaluation out-of-sample (TE, Sharpe, drawdown, hit ratio)
    5. Visualisation et rapport complet
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)

INDEX_NAME      = "CAC 40 (Synthétique)"
N_STOCKS        = 40        # taille de l'univers
BASKET_SIZES    = [5, 10, 15, 20]  # tailles de panier à comparer
N_DAYS_TRAIN    = 504       # 2 ans d'in-sample
N_DAYS_TEST     = 252       # 1 an d'out-of-sample
TRADING_DAYS    = 252
REBAL_FREQ      = 63        # rebalancement trimestriel (jours)

# Capitalisation boursière fictive des 40 titres (pour pondération indice)
SECTORS = {
    "Énergie":       ["TTE", "ENI"],
    "Financières":   ["BNP", "ACA", "GLE", "AXA"],
    "Industrie":     ["AIR", "SAF", "LG", "TKA", "SCH"],
    "Consommation":  ["MC", "KER", "RNO", "STM", "RI"],
    "Santé":         ["SAN", "EL", "DBK"],
    "Technologie":   ["CAP", "ATO", "ORA"],
    "Services":      ["VIE", "DG", "PUB", "DSY"],
    "Immobilier":    ["URW", "ICB"],
    "Matériaux":     ["ARG", "SOLB"],
    "Utilities":     ["EDF", "VGB", "DTE"],
    "Alimentation":  ["DAN", "NESN", "UNI", "COL", "BON", "KLN"],
}

TICKERS = [t for tickers in SECTORS.values() for t in tickers][:N_STOCKS]
N_STOCKS = len(TICKERS)

# ─────────────────────────────────────────────────────────────────────────────
# 1. GÉNÉRATION DE DONNÉES SYNTHÉTIQUES — MODÈLE FACTORIEL
# ─────────────────────────────────────────────────────────────────────────────

def generate_factor_model_data(n_stocks: int, n_days: int, tickers: list) -> tuple:
    """
    Génère des rendements synthétiques réalistes via un modèle factoriel à 3 facteurs :
      - Facteur marché (beta ~ 1, vol 15%)
      - Facteur sectoriel (vol 8%)
      - Facteur idiosyncratique (vol 10-20% selon le titre)
    Reproduit les statistiques typiques d'actions européennes.
    """
    n_total = n_days

    # Facteur marché : mu annuel ~8%, vol ~15%
    mu_market  = 0.08 / TRADING_DAYS
    vol_market = 0.15 / np.sqrt(TRADING_DAYS)
    market_factor = np.random.normal(mu_market, vol_market, n_total)

    # Facteur sectoriel (3 secteurs distincts)
    n_sectors = 3
    sector_factors = np.random.normal(0, 0.08 / np.sqrt(TRADING_DAYS),
                                      (n_sectors, n_total))
    sector_assignment = np.array([i % n_sectors for i in range(n_stocks)])

    # Betas marché (0.7 à 1.3), betas sectoriels (0 à 0.5)
    betas_market  = np.random.uniform(0.7, 1.3, n_stocks)
    betas_sector  = np.random.uniform(0.0, 0.5, n_stocks)

    # Volatilité idiosyncratique (10% à 20% annualisée)
    vols_idio = np.random.uniform(0.10, 0.20, n_stocks) / np.sqrt(TRADING_DAYS)

    # Rendements actions
    returns_matrix = np.zeros((n_total, n_stocks))
    for i in range(n_stocks):
        idio = np.random.normal(0, vols_idio[i], n_total)
        sec  = sector_factors[sector_assignment[i]]
        returns_matrix[:, i] = (betas_market[i] * market_factor
                                 + betas_sector[i] * sec
                                 + idio)

    # Capitalisations boursières fictives (log-normale, milliards €)
    market_caps = np.exp(np.random.normal(3.5, 0.8, n_stocks))
    index_weights = market_caps / market_caps.sum()

    # Rendement de l'indice = somme pondérée
    index_returns = returns_matrix @ index_weights

    # Conversion en prix (base 100)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns_matrix, axis=0)),
        columns=tickers
    )
    index_prices = pd.Series(
        100 * np.exp(np.cumsum(index_returns)),
        name=INDEX_NAME
    )

    returns_df = pd.DataFrame(returns_matrix, columns=tickers)
    index_ret   = pd.Series(index_returns, name=INDEX_NAME)

    return prices, returns_df, index_prices, index_ret, index_weights


# ─────────────────────────────────────────────────────────────────────────────
# 2. MÉTRIQUES DE RISQUE
# ─────────────────────────────────────────────────────────────────────────────

def tracking_error(portfolio_returns: np.ndarray,
                   index_returns: np.ndarray,
                   annualize: bool = True) -> float:
    """Tracking error annualisée (écart-type des différences de rendements)."""
    diff = portfolio_returns - index_returns
    te = np.std(diff, ddof=1)
    return te * np.sqrt(TRADING_DAYS) if annualize else te


def information_ratio(portfolio_returns: np.ndarray,
                      index_returns: np.ndarray) -> float:
    """Information Ratio = alpha annualisé / TE annualisée."""
    diff = portfolio_returns - index_returns
    alpha = np.mean(diff) * TRADING_DAYS
    te    = tracking_error(portfolio_returns, index_returns)
    return alpha / te if te > 0 else np.nan


def max_drawdown(cum_returns: np.ndarray) -> float:
    """Maximum drawdown sur une série de rendements cumulés."""
    if len(cum_returns) == 0:
        return np.nan
    running_max = np.maximum.accumulate(cum_returns)
    drawdown    = (cum_returns - running_max) / running_max
    return float(np.min(drawdown))


def sharpe_ratio(returns: np.ndarray, rf: float = 0.03) -> float:
    """Sharpe ratio annualisé (taux sans risque = 3%)."""
    mu  = np.mean(returns) * TRADING_DAYS
    vol = np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS)
    return (mu - rf) / vol if vol > 0 else np.nan


def var_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    """VaR historique (perte max au seuil de confiance donné)."""
    return -np.percentile(returns, (1 - confidence) * 100)


def cvar_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    """CVaR (Expected Shortfall) historique."""
    threshold = -var_historical(returns, confidence)
    tail = returns[returns <= threshold]
    return -np.mean(tail) if len(tail) > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# 3. SÉLECTION DU PANIER — GREEDY PAR CORRÉLATION
# ─────────────────────────────────────────────────────────────────────────────

def greedy_basket_selection(returns_df: pd.DataFrame,
                             index_returns: pd.Series,
                             basket_size: int) -> list:
    """
    Sélection gloutonne des titres les plus corrélés à l'indice.
    À chaque étape, on ajoute le titre qui maximise la corrélation
    du rendement marginal avec l'erreur de réplication résiduelle.

    Cette approche reproduit la logique utilisée en gestion passive
    pour la sélection de paniers de réplication partielle.
    """
    selected  = []
    residual  = index_returns.values.copy()
    candidates = list(returns_df.columns)

    for _ in range(basket_size):
        best_stock = None
        best_corr  = -np.inf

        for stock in candidates:
            if stock in selected:
                continue
            r = returns_df[stock].values
            # Corrélation du titre avec l'erreur résiduelle courante
            corr = np.corrcoef(r, residual)[0, 1]
            if corr > best_corr:
                best_corr  = corr
                best_stock = stock

        if best_stock:
            selected.append(best_stock)
            # Mise à jour résiduelle (retrait de la contribution du titre sélectionné)
            r_best = returns_df[best_stock].values
            beta   = np.cov(r_best, residual)[0, 1] / np.var(r_best)
            residual = residual - beta * r_best

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# 4. OPTIMISATION DES POIDS — MINIMISATION TE (SLSQP)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_weights(returns_basket: np.ndarray,
                     index_returns: np.ndarray,
                     max_weight: float = 0.30) -> np.ndarray:
    """
    Optimisation quadratique des poids du panier de réplication.

    Problème :  min  TE(w)  =  Var( R_basket @ w  -  R_index )
    Contraintes :
      - sum(w) = 1        (budget)
      - w_i >= 0          (long only)
      - w_i <= max_weight (concentration maximale)

    Solveur : SLSQP (Sequential Least-Squares Programming, SciPy)
    Mathématiquement équivalent à cvxpy avec OSQP sur ce problème.
    """
    n = returns_basket.shape[1]
    w0 = np.ones(n) / n  # initialisation équipondérée

    def objective(w):
        portfolio_ret = returns_basket @ w
        diff = portfolio_ret - index_returns
        return np.var(diff, ddof=1) * TRADING_DAYS  # TE² annualisée

    def objective_grad(w):
        """Gradient analytique de la TE² pour accélérer la convergence."""
        n_obs = len(index_returns)
        portfolio_ret = returns_basket @ w
        diff = portfolio_ret - index_returns
        mean_diff = np.mean(diff)
        # ∂(TE²)/∂w = 2/T * Σ [(r_portfolio,t - r_index,t - mean_diff) * r_basket,t]
        grad = 2 * TRADING_DAYS / n_obs * returns_basket.T @ (diff - mean_diff)
        return grad

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n

    result = minimize(
        objective,
        w0,
        jac=objective_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000}
    )

    if not result.success:
        # Fallback : pondération par corrélation à l'indice
        corrs = np.array([abs(np.corrcoef(returns_basket[:, i], index_returns)[0, 1])
                          for i in range(n)])
        w_fallback = corrs / corrs.sum()
        return np.clip(w_fallback, 0, max_weight) / np.clip(w_fallback, 0, max_weight).sum()

    w_opt = result.x
    w_opt = np.clip(w_opt, 0, max_weight)
    return w_opt / w_opt.sum()


# ─────────────────────────────────────────────────────────────────────────────
# 5. BACKTESTING AVEC REBALANCEMENT DYNAMIQUE
# ─────────────────────────────────────────────────────────────────────────────

def backtest_with_rebalancing(returns_df: pd.DataFrame,
                               index_returns: pd.Series,
                               basket_size: int,
                               train_window: int = N_DAYS_TRAIN,
                               rebal_freq: int = REBAL_FREQ) -> dict:
    """
    Backtest walk-forward avec rebalancement trimestriel.
    À chaque date de rebalancement :
      1. Sélection greedy du panier sur la fenêtre roulante d'entraînement
      2. Optimisation des poids
      3. Application sur la période suivante jusqu'au prochain rebalancement
    """
    all_portfolio_ret = []
    all_index_ret     = []
    rebalance_dates   = []
    weight_history    = {}

    n_total = len(returns_df)
    t = train_window

    while t < n_total:
        # Fenêtre d'entraînement
        r_train  = returns_df.iloc[t - train_window:t]
        ri_train = index_returns.iloc[t - train_window:t]

        # Sélection et optimisation
        basket = greedy_basket_selection(r_train, ri_train, basket_size)
        w_opt  = optimize_weights(r_train[basket].values, ri_train.values)

        # Application sur la prochaine période
        end = min(t + rebal_freq, n_total)
        r_test  = returns_df.iloc[t:end][basket].values
        ri_test = index_returns.iloc[t:end].values

        port_ret = r_test @ w_opt
        all_portfolio_ret.extend(port_ret.tolist())
        all_index_ret.extend(ri_test.tolist())
        rebalance_dates.append(t)
        weight_history[t] = dict(zip(basket, w_opt))

        t = end

    return {
        "portfolio_returns": np.array(all_portfolio_ret),
        "index_returns":     np.array(all_index_ret),
        "rebalance_dates":   rebalance_dates,
        "weight_history":    weight_history,
        "basket_size":       basket_size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. RAPPORT DE PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance_report(result: dict) -> dict:
    """Calcule toutes les métriques de performance et de risque."""
    pr = result["portfolio_returns"]
    ir = result["index_returns"]

    cum_portfolio = np.exp(np.cumsum(pr))
    cum_index     = np.exp(np.cumsum(ir))

    te = tracking_error(pr, ir)
    n  = len(pr)

    # Hit ratio : % de jours où le portefeuille surperforme l'indice
    hit_ratio = np.mean(pr > ir)

    return {
        "Tracking Error Annualisée (%)":     round(te * 100, 4),
        "Information Ratio":                  round(information_ratio(pr, ir), 4),
        "Sharpe Ratio (Portefeuille)":        round(sharpe_ratio(pr), 4),
        "Sharpe Ratio (Indice)":              round(sharpe_ratio(ir), 4),
        "Rendement Annualisé Port. (%)":      round(np.mean(pr) * TRADING_DAYS * 100, 4),
        "Rendement Annualisé Indice (%)":     round(np.mean(ir) * TRADING_DAYS * 100, 4),
        "Volatilité Annualisée Port. (%)":    round(np.std(pr) * np.sqrt(TRADING_DAYS) * 100, 4),
        "Max Drawdown Port. (%)":             round(max_drawdown(cum_portfolio) * 100, 4),
        "Max Drawdown Indice (%)":            round(max_drawdown(cum_index) * 100, 4),
        "VaR 95% Journalière (%)":            round(var_historical(pr, 0.95) * 100, 4),
        "CVaR 95% Journalière (%)":           round(cvar_historical(pr, 0.95) * 100, 4),
        "Hit Ratio (%)":                      round(hit_ratio * 100, 2),
        "Nombre de Titres":                   result["basket_size"],
        "Nombre de Rebalancements":           len(result["rebalance_dates"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATION — DASHBOARD 4 PANNEAUX
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(results_by_size: dict, index_returns: pd.Series):
    """
    Dashboard complet en 4 panneaux :
      [A] Performance cumulée : panier vs indice (par taille)
      [B] Tracking Error rolling 60j par taille de panier
      [C] Distribution des erreurs de réplication quotidiennes
      [D] Tableau comparatif des métriques de risque
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig    = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f1117")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # ── Panneau A : Performance cumulée ──────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor("#1a1d23")
    cum_index = np.exp(np.cumsum(list(results_by_size.values())[0]["index_returns"])) * 100

    ax_a.plot(cum_index, color="white", linewidth=2.2, label=INDEX_NAME, linestyle="--", alpha=0.9)
    for i, (size, res) in enumerate(results_by_size.items()):
        cum_port = np.exp(np.cumsum(res["portfolio_returns"])) * 100
        ax_a.plot(cum_port, color=colors[i], linewidth=1.6, label=f"Panier {size} titres", alpha=0.85)

    ax_a.set_title("Performance Cumulée — Panier vs Indice", color="white", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Valeur (base 100)", color="#aaaaaa", fontsize=9)
    ax_a.set_xlabel("Jours (out-of-sample)", color="#aaaaaa", fontsize=9)
    ax_a.tick_params(colors="#aaaaaa")
    ax_a.spines[:].set_color("#333333")
    leg_a = ax_a.legend(fontsize=8, loc="upper left")
    for text in leg_a.get_texts():
        text.set_color("white")
    leg_a.get_frame().set_facecolor("#2a2d33")
    ax_a.yaxis.label.set_color("#aaaaaa")

    # ── Panneau B : Tracking Error Rolling 60j ───────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor("#1a1d23")
    window = 60

    for i, (size, res) in enumerate(results_by_size.items()):
        pr = res["portfolio_returns"]
        ir = res["index_returns"]
        diff = pr - ir
        te_rolling = [
            np.std(diff[max(0, j - window):j], ddof=1) * np.sqrt(TRADING_DAYS) * 100
            for j in range(window, len(diff))
        ]
        ax_b.plot(te_rolling, color=colors[i], linewidth=1.4,
                  label=f"Panier {size} titres", alpha=0.85)

    ax_b.set_title(f"Tracking Error Rolling {window}j (Annualisée, %)",
                   color="white", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("TE Annualisée (%)", color="#aaaaaa", fontsize=9)
    ax_b.set_xlabel("Jours", color="#aaaaaa", fontsize=9)
    ax_b.tick_params(colors="#aaaaaa")
    ax_b.spines[:].set_color("#333333")
    leg_b = ax_b.legend(fontsize=8)
    for text in leg_b.get_texts():
        text.set_color("white")
    leg_b.get_frame().set_facecolor("#2a2d33")

    # ── Panneau C : Distribution des erreurs de réplication ──────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor("#1a1d23")

    for i, (size, res) in enumerate(results_by_size.items()):
        diff = (res["portfolio_returns"] - res["index_returns"]) * 100
        ax_c.hist(diff, bins=40, alpha=0.45, color=colors[i],
                  label=f"{size} titres", density=True, edgecolor="none")

    # Courbe normale de référence
    x_ref = np.linspace(-2.5, 2.5, 200)
    ax_c.plot(x_ref, norm.pdf(x_ref, 0, 0.5), color="white",
              linewidth=1.5, linestyle=":", alpha=0.6, label="Normale (ref)")

    ax_c.axvline(0, color="#ff4444", linewidth=1.2, linestyle="--", alpha=0.7)
    ax_c.set_title("Distribution des Erreurs de Réplication Quotidiennes (%)",
                   color="white", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Erreur de Réplication (%)", color="#aaaaaa", fontsize=9)
    ax_c.set_ylabel("Densité", color="#aaaaaa", fontsize=9)
    ax_c.tick_params(colors="#aaaaaa")
    ax_c.spines[:].set_color("#333333")
    leg_c = ax_c.legend(fontsize=8)
    for text in leg_c.get_texts():
        text.set_color("white")
    leg_c.get_frame().set_facecolor("#2a2d33")

    # ── Panneau D : Tableau comparatif ───────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor("#1a1d23")
    ax_d.axis("off")

    metrics_to_show = [
        "Tracking Error Annualisée (%)",
        "Information Ratio",
        "Sharpe Ratio (Portefeuille)",
        "Rendement Annualisé Port. (%)",
        "Max Drawdown Port. (%)",
        "VaR 95% Journalière (%)",
        "CVaR 95% Journalière (%)",
        "Hit Ratio (%)",
    ]
    col_labels = ["Métrique"] + [f"{s} titres" for s in results_by_size.keys()]
    table_data = []
    for metric in metrics_to_show:
        row = [metric] + [str(compute_performance_report(res)[metric])
                          for res in results_by_size.values()]
        table_data.append(row)

    table = ax_d.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("#1a1d23" if row % 2 == 0 else "#22262e")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#333333")
        if row == 0:
            cell.set_facecolor("#2c5f9e")
            cell.set_text_props(color="white", fontweight="bold")

    ax_d.set_title("Tableau Comparatif des Métriques de Risque",
                   color="white", fontsize=11, fontweight="bold", pad=10)

    # ── Titre global ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"Index Replication Optimizer — {INDEX_NAME}\n"
        f"In-sample : {N_DAYS_TRAIN}j  |  Out-of-sample : {N_DAYS_TEST}j  |  "
        f"Rebalancement : {REBAL_FREQ}j  |  Contrainte max poids : 30%",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    plt.savefig("/mnt/user-data/outputs/index_replication_dashboard.png",
                dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("✅  Dashboard sauvegardé.")


# ─────────────────────────────────────────────────────────────────────────────
# 8. RAPPORT CONSOLE
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results_by_size: dict):
    sep = "─" * 72
    print(f"\n{'═'*72}")
    print(f"  INDEX REPLICATION OPTIMIZER — RAPPORT DE PERFORMANCE")
    print(f"  Univers : {N_STOCKS} titres ({INDEX_NAME}) | Train : {N_DAYS_TRAIN}j | Test : {N_DAYS_TEST}j")
    print(f"{'═'*72}\n")

    for size, res in results_by_size.items():
        report = compute_performance_report(res)
        print(f"{'┌' + sep[1:]}")
        print(f"│  PANIER DE {size} TITRES — Réplication partielle avec rebalancement trimestriel")
        print(f"{'├' + sep[1:]}")
        for k, v in report.items():
            print(f"│  {k:<42} {str(v):>12}")
        print(f"{'└' + sep[1:]}\n")

    # Sélection du meilleur panier (min TE)
    best_size = min(results_by_size.keys(),
                    key=lambda s: compute_performance_report(results_by_size[s])["Tracking Error Annualisée (%)"])
    best_te   = compute_performance_report(results_by_size[best_size])["Tracking Error Annualisée (%)"]
    print(f"  ▶ Meilleur panier (min TE) : {best_size} titres | TE = {best_te}%")
    print(f"  ▶ Seuil réglementaire indicatif (UCITS) : TE < 1.5% pour ETF indiciels")

    # Affichage de la composition du dernier rebalancement (best basket)
    res_best = results_by_size[best_size]
    last_rebal = res_best["rebalance_dates"][-1]
    last_weights = res_best["weight_history"][last_rebal]
    print(f"\n  Composition du panier {best_size} titres (dernier rebalancement — jour {last_rebal}) :")
    for stock, w in sorted(last_weights.items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 100 // 2)
        print(f"    {stock:<6}  {w*100:5.2f}%  {bar}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("⏳  Génération des données synthétiques (modèle factoriel 3 facteurs)...")
    n_total = N_DAYS_TRAIN + N_DAYS_TEST
    prices, returns_df, index_prices, index_ret, index_weights = \
        generate_factor_model_data(N_STOCKS, n_total, TICKERS)

    # Walk-forward sur données complètes (train = N_DAYS_TRAIN, test = période suivante)
    print(f"⏳  Backtesting walk-forward pour {len(BASKET_SIZES)} tailles de panier...")
    results_by_size = {}
    for size in BASKET_SIZES:
        print(f"    → Panier {size} titres...", end=" ")
        res = backtest_with_rebalancing(
            returns_df,        # données complètes
            index_ret,
            basket_size=size,
            train_window=N_DAYS_TRAIN,
            rebal_freq=REBAL_FREQ
        )
        results_by_size[size] = res
        te = tracking_error(res["portfolio_returns"], res["index_returns"])
        print(f"TE = {te*100:.4f}%")

    print_report(results_by_size)

    print("⏳  Génération du dashboard...")
    plot_dashboard(results_by_size, index_ret.iloc[N_DAYS_TRAIN:])
    print("\n✅  Projet complet. Fichier : index_replication_dashboard.png")


if __name__ == "__main__":
    main()
