"""
=============================================================================
  Index Replication Optimizer — Interface Streamlit
  Auteur : Mohamed Amine El Aouadi
  Lancer : streamlit run app.py
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Index Replication Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS dark theme
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: #1a1d23;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stMetric { background: #1a1d23; border-radius: 8px; padding: 12px; }
    .badge-ok  { color: #00cc88; font-weight: bold; }
    .badge-warn{ color: #ff9944; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

TRADING_DAYS = 252
TICKERS = [
    "TTE","ENI","BNP","ACA","GLE","AXA","AIR","SAF","LG","TKA",
    "SCH","MC","KER","RNO","STM","RI","SAN","EL","DBK","CAP",
    "ATO","ORA","VIE","DG","PUB","DSY","URW","ICB","ARG","SOLB",
    "EDF","VGB","DTE","DAN","NESN","UNI","COL","BON","KLN"
]
N_STOCKS = len(TICKERS)

# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS (identiques à index_replication.py)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def generate_data(n_days: int, seed: int = 42) -> tuple:
    np.random.seed(seed)
    mu_market  = 0.08 / TRADING_DAYS
    vol_market = 0.15 / np.sqrt(TRADING_DAYS)
    market_factor = np.random.normal(mu_market, vol_market, n_days)
    n_sectors = 3
    sector_factors = np.random.normal(0, 0.08/np.sqrt(TRADING_DAYS), (n_sectors, n_days))
    sector_assignment = np.array([i % n_sectors for i in range(N_STOCKS)])
    betas_market = np.random.uniform(0.7, 1.3, N_STOCKS)
    betas_sector = np.random.uniform(0.0, 0.5, N_STOCKS)
    vols_idio    = np.random.uniform(0.10, 0.20, N_STOCKS) / np.sqrt(TRADING_DAYS)
    returns_matrix = np.zeros((n_days, N_STOCKS))
    for i in range(N_STOCKS):
        idio = np.random.normal(0, vols_idio[i], n_days)
        sec  = sector_factors[sector_assignment[i]]
        returns_matrix[:, i] = betas_market[i]*market_factor + betas_sector[i]*sec + idio
    market_caps   = np.exp(np.random.normal(3.5, 0.8, N_STOCKS))
    index_weights = market_caps / market_caps.sum()
    index_returns = returns_matrix @ index_weights
    returns_df = pd.DataFrame(returns_matrix, columns=TICKERS)
    index_ret  = pd.Series(index_returns, name="CAC 40")
    return returns_df, index_ret


def greedy_selection(returns_df, index_returns, basket_size):
    selected  = []
    residual  = index_returns.values.copy()
    candidates = list(returns_df.columns)
    for _ in range(basket_size):
        best_stock, best_corr = None, -np.inf
        for stock in candidates:
            if stock in selected:
                continue
            r    = returns_df[stock].values
            corr = np.corrcoef(r, residual)[0, 1]
            if corr > best_corr:
                best_corr, best_stock = corr, stock
        if best_stock:
            selected.append(best_stock)
            r_best   = returns_df[best_stock].values
            beta     = np.cov(r_best, residual)[0,1] / np.var(r_best)
            residual = residual - beta * r_best
    return selected


def optimize_weights(returns_basket, index_returns, max_weight):
    n  = returns_basket.shape[1]
    w0 = np.ones(n) / n

    def objective(w):
        diff = returns_basket @ w - index_returns
        return np.var(diff, ddof=1) * TRADING_DAYS

    def grad(w):
        n_obs = len(index_returns)
        diff  = returns_basket @ w - index_returns
        return 2 * TRADING_DAYS / n_obs * returns_basket.T @ (diff - np.mean(diff))

    result = minimize(objective, w0, jac=grad, method="SLSQP",
                      bounds=[(0, max_weight)]*n,
                      constraints=[{"type":"eq","fun":lambda w: w.sum()-1}],
                      options={"ftol":1e-12,"maxiter":1000})
    w = np.clip(result.x if result.success else w0, 0, max_weight)
    return w / w.sum()


def backtest(returns_df, index_ret, basket_size, train_window, rebal_freq, max_weight):
    all_pr, all_ir = [], []
    weight_history, rebal_dates = {}, []
    n_total = len(returns_df)
    t = train_window
    while t < n_total:
        r_train  = returns_df.iloc[t-train_window:t]
        ri_train = index_ret.iloc[t-train_window:t]
        basket   = greedy_selection(r_train, ri_train, basket_size)
        w_opt    = optimize_weights(r_train[basket].values, ri_train.values, max_weight)
        end      = min(t + rebal_freq, n_total)
        r_test   = returns_df.iloc[t:end][basket].values
        ri_test  = index_ret.iloc[t:end].values
        all_pr.extend((r_test @ w_opt).tolist())
        all_ir.extend(ri_test.tolist())
        rebal_dates.append(t)
        weight_history[t] = dict(zip(basket, w_opt))
        t = end
    return {
        "pr": np.array(all_pr), "ir": np.array(all_ir),
        "weight_history": weight_history, "rebal_dates": rebal_dates
    }


def metrics(pr, ir):
    te   = np.std(pr - ir, ddof=1) * np.sqrt(TRADING_DAYS)
    diff = pr - ir
    alpha = np.mean(diff) * TRADING_DAYS
    ir_   = alpha / te if te > 0 else np.nan
    sharpe_p = (np.mean(pr)*TRADING_DAYS - 0.03) / (np.std(pr)*np.sqrt(TRADING_DAYS))
    sharpe_i = (np.mean(ir)*TRADING_DAYS - 0.03) / (np.std(ir)*np.sqrt(TRADING_DAYS))
    cum_p = np.exp(np.cumsum(pr))
    mdd_p = float(np.min((cum_p - np.maximum.accumulate(cum_p)) / np.maximum.accumulate(cum_p)))
    var95 = -np.percentile(pr, 5)
    cvar95 = -np.mean(pr[pr <= -var95])
    return {
        "te": te, "ir": ir_, "sharpe_p": sharpe_p, "sharpe_i": sharpe_i,
        "ret_p": np.mean(pr)*TRADING_DAYS, "ret_i": np.mean(ir)*TRADING_DAYS,
        "vol_p": np.std(pr)*np.sqrt(TRADING_DAYS),
        "mdd_p": mdd_p, "var95": var95, "cvar95": cvar95,
        "hit": np.mean(pr > ir)
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 📊 Index Replication Optimizer")
st.markdown("**Réplication indicielle partielle — CAC 40 Synthétique** | Optimisation SLSQP sous contraintes | Walk-forward backtesting")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Paramètres")

    basket_size   = st.slider("Nombre de titres dans le panier", 5, 30, 15, 1)
    max_weight_pct= st.slider("Poids maximum par titre (%)", 10, 40, 30, 5)
    max_weight    = max_weight_pct / 100

    rebal_label   = st.selectbox("Fréquence de rebalancement",
                                  ["Mensuel (21j)", "Trimestriel (63j)", "Semestriel (126j)"],
                                  index=1)
    rebal_freq    = {"Mensuel (21j)": 21, "Trimestriel (63j)": 63, "Semestriel (126j)": 126}[rebal_label]

    train_window  = st.slider("Fenêtre d'entraînement (jours)", 126, 756, 504, 63)

    compare_mode  = st.checkbox("Comparer plusieurs tailles de panier", value=False)
    if compare_mode:
        sizes_to_compare = st.multiselect("Tailles à comparer", [5,10,15,20,25,30],
                                           default=[5,10,15,20])

    st.divider()
    st.caption("ℹ️ Données synthétiques générées par modèle factoriel 3 facteurs (marché, sectoriel, idiosyncratique). Résultats illustratifs.")

    run = st.button("🚀 Lancer l'optimisation", type="primary", use_container_width=True)

# ── Main content ──────────────────────────────────────────────────────────────
if not run:
    st.info("👈 Configure les paramètres dans le panneau latéral, puis clique sur **Lancer l'optimisation**.")

    with st.expander("📖 Comment fonctionne ce projet ?"):
        st.markdown("""
        Ce projet implémente un **optimiseur de réplication indicielle partielle** — technique utilisée en gestion ETF pour 
        répliquer un indice boursier avec un sous-ensemble de titres (panier), réduisant ainsi les coûts de transaction 
        tout en maintenant une tracking error minimale.
        
        **Pipeline en 4 étapes :**
        1. **Modèle factoriel** — Génération de rendements synthétiques réalistes via 3 facteurs (marché, sectoriel, idiosyncratique)
        2. **Sélection gloutonne** — Identification des titres les plus corrélés à l'erreur de réplication résiduelle
        3. **Optimisation SLSQP** — Minimisation de la Tracking Error² sous contraintes (budget, long-only, concentration max)
        4. **Backtest walk-forward** — Évaluation out-of-sample avec rebalancement dynamique
        
        **Métriques clés analysées :** Tracking Error annualisée, Information Ratio, Sharpe Ratio, VaR/CVaR 95%, Maximum Drawdown, Hit Ratio
        
        **Seuil réglementaire de référence :** TE < 1.5% pour qualification UCITS ETF indiciel
        """)
    st.stop()

# ── Exécution ─────────────────────────────────────────────────────────────────
n_total = train_window + 252
with st.spinner("Génération des données synthétiques..."):
    returns_df, index_ret = generate_data(n_total)

if compare_mode and sizes_to_compare:
    target_sizes = sizes_to_compare
else:
    target_sizes = [basket_size]

results = {}
prog = st.progress(0, text="Optimisation en cours...")
for i, size in enumerate(target_sizes):
    prog.progress((i+1)/len(target_sizes), text=f"Panier {size} titres...")
    res = backtest(returns_df, index_ret, size, train_window, rebal_freq, max_weight)
    results[size] = res
prog.empty()

# ── Métriques principales (panier principal) ──────────────────────────────────
main_size = target_sizes[0] if not compare_mode else target_sizes[0]
m = metrics(results[main_size]["pr"], results[main_size]["ir"])
te_pct = m["te"] * 100
ucits_ok = te_pct < 1.5

st.subheader("📌 Métriques Clés" + (f" — Panier {main_size} titres" if not compare_mode else ""))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tracking Error",
          f"{te_pct:.2f}%",
          delta="✅ < 1.5% UCITS" if ucits_ok else "⚠️ > seuil UCITS",
          delta_color="normal" if ucits_ok else "inverse")
c2.metric("Information Ratio",   f"{m['ir']:.3f}")
c3.metric("Sharpe (Portefeuille)", f"{m['sharpe_p']:.3f}")
c4.metric("Sharpe (Indice)",      f"{m['sharpe_i']:.3f}")
c5.metric("Hit Ratio",            f"{m['hit']*100:.1f}%")

c6, c7, c8, c9 = st.columns(4)
c6.metric("Rendement Annualisé", f"{m['ret_p']*100:.2f}%")
c7.metric("Volatilité Annualisée", f"{m['vol_p']*100:.2f}%")
c8.metric("Max Drawdown",         f"{m['mdd_p']*100:.2f}%")
c9.metric("CVaR 95%",             f"{m['cvar95']*100:.2f}%")

st.divider()

# ── Graphiques ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

BG    = "#0f1117"
BG2   = "#1a1d23"
COLORS = ["#4c9be8", "#ff7f40", "#2ecc71", "#e74c3c", "#9b59b6", "#f1c40f"]

with col_left:
    st.subheader("📈 Performance Cumulée")
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)

    # Indice
    ir_main = results[list(results.keys())[0]]["ir"]
    cum_idx = np.exp(np.cumsum(ir_main)) * 100
    ax.plot(cum_idx, color="white", lw=2.2, ls="--", label="CAC 40 (indice)", alpha=0.9)

    for i, (size, res) in enumerate(results.items()):
        cum_p = np.exp(np.cumsum(res["pr"])) * 100
        ax.plot(cum_p, color=COLORS[i], lw=1.8, label=f"Panier {size} titres", alpha=0.9)

    ax.tick_params(colors="#aaa"); ax.spines[:].set_color("#333")
    ax.set_xlabel("Jours (out-of-sample)", color="#aaa", fontsize=9)
    ax.set_ylabel("Valeur (base 100)", color="#aaa", fontsize=9)
    leg = ax.legend(fontsize=8, facecolor="#2a2d33", labelcolor="white")
    st.pyplot(fig, use_container_width=True)

with col_right:
    st.subheader("📉 Tracking Error Rolling 60j")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.patch.set_facecolor(BG); ax2.set_facecolor(BG2)
    window = 60

    for i, (size, res) in enumerate(results.items()):
        diff = res["pr"] - res["ir"]
        te_roll = [np.std(diff[max(0,j-window):j], ddof=1)*np.sqrt(TRADING_DAYS)*100
                   for j in range(window, len(diff))]
        ax2.plot(te_roll, color=COLORS[i], lw=1.6, label=f"Panier {size} titres", alpha=0.9)

    ax2.axhline(1.5, color="#ff4444", lw=1.4, ls=":", alpha=0.8, label="Seuil UCITS 1.5%")
    ax2.tick_params(colors="#aaa"); ax2.spines[:].set_color("#333")
    ax2.set_xlabel("Jours", color="#aaa", fontsize=9)
    ax2.set_ylabel("TE annualisée (%)", color="#aaa", fontsize=9)
    leg2 = ax2.legend(fontsize=8, facecolor="#2a2d33", labelcolor="white")
    st.pyplot(fig2, use_container_width=True)

# ── Distribution des erreurs ──────────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("📊 Distribution des Erreurs de Réplication")
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    fig3.patch.set_facecolor(BG); ax3.set_facecolor(BG2)

    for i, (size, res) in enumerate(results.items()):
        diff = (res["pr"] - res["ir"]) * 100
        ax3.hist(diff, bins=40, alpha=0.45, color=COLORS[i],
                 label=f"{size} titres", density=True, edgecolor="none")

    x_ref = np.linspace(-2.5, 2.5, 200)
    ax3.plot(x_ref, norm.pdf(x_ref, 0, 0.5), color="white", lw=1.4, ls=":", alpha=0.6, label="Normale")
    ax3.axvline(0, color="#ff4444", lw=1.2, ls="--", alpha=0.7)
    ax3.tick_params(colors="#aaa"); ax3.spines[:].set_color("#333")
    ax3.set_xlabel("Erreur journalière (%)", color="#aaa", fontsize=9)
    ax3.set_ylabel("Densité", color="#aaa", fontsize=9)
    leg3 = ax3.legend(fontsize=8, facecolor="#2a2d33", labelcolor="white")
    st.pyplot(fig3, use_container_width=True)

with col4:
    st.subheader("🏗️ Composition du Panier (dernier rebalancement)")
    res_main = results[main_size]
    last_w   = res_main["weight_history"][res_main["rebal_dates"][-1]]
    df_w = pd.DataFrame({
        "Titre":     list(last_w.keys()),
        "Poids (%)": [round(v*100, 2) for v in last_w.values()]
    }).sort_values("Poids (%)", ascending=False).reset_index(drop=True)

    fig4, ax4 = plt.subplots(figsize=(7, 4))
    fig4.patch.set_facecolor(BG); ax4.set_facecolor(BG2)
    bars = ax4.barh(df_w["Titre"], df_w["Poids (%)"], color=COLORS[0], alpha=0.85, edgecolor="none")
    ax4.axvline(max_weight_pct, color="#ff4444", lw=1.2, ls="--", alpha=0.7,
                label=f"Max {max_weight_pct}%")
    ax4.invert_yaxis()
    ax4.tick_params(colors="#aaa", labelsize=8); ax4.spines[:].set_color("#333")
    ax4.set_xlabel("Poids (%)", color="#aaa", fontsize=9)
    leg4 = ax4.legend(fontsize=8, facecolor="#2a2d33", labelcolor="white")
    st.pyplot(fig4, use_container_width=True)

# ── Tableau comparatif (si compare_mode) ──────────────────────────────────────
if compare_mode and len(target_sizes) > 1:
    st.divider()
    st.subheader("📋 Tableau Comparatif")
    rows = []
    for size, res in results.items():
        m_ = metrics(res["pr"], res["ir"])
        rows.append({
            "Taille panier": size,
            "TE annualisée (%)":      round(m_["te"]*100, 4),
            "Information Ratio":       round(m_["ir"], 4),
            "Sharpe Ratio":            round(m_["sharpe_p"], 4),
            "Rendement annualisé (%)": round(m_["ret_p"]*100, 4),
            "Max Drawdown (%)":        round(m_["mdd_p"]*100, 4),
            "VaR 95% (%)":             round(m_["var95"]*100, 4),
            "CVaR 95% (%)":            round(m_["cvar95"]*100, 4),
            "Hit Ratio (%)":           round(m_["hit"]*100, 2),
            "UCITS OK ?":              "✅" if m_["te"]*100 < 1.5 else "❌",
        })
    df_compare = pd.DataFrame(rows).set_index("Taille panier")
    st.dataframe(df_compare, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Mohamed Amine El Aouadi** — MSc Big Data & Finance Quantitative, NEOMA Business School (2026–2027) | "
    "Candidature : Alternance Front Office Risk Officer ETF/Index, BNP Paribas Asset Management"
)
