# 📊 Index Replication Optimizer

> **Réplication indicielle partielle (CAC 40) — Optimisation quadratique sous contraintes, backtesting walk-forward**  
> *Projet Python personnel — Mohamed Amine El Aouadi | MSc Big Data & Finance Quantitative, NEOMA Business School*

---

## 🎯 Objectif

Répliquer un indice boursier (CAC 40 synthétique, 39 titres) avec un **sous-ensemble de N titres (panier)**, en minimisant la **Tracking Error ex-ante** par optimisation quadratique sous contraintes.

Ce problème est au cœur de la **gestion indicielle et des ETFs** : chaque titre supplémentaire dans le panier réduit l'erreur de réplication mais augmente les coûts de transaction et la complexité opérationnelle. Ce projet modélise ce compromis et identifie le seuil optimal.

---

## 🏗️ Architecture

```
index-replication-optimizer/
├── index_replication.py   # Script principal (CLI + visualisation statique)
├── app.py                 # Interface Streamlit interactive
├── requirements.txt       # Dépendances
└── output/
    └── dashboard.png      # Résultats du backtest
```

---

## ⚙️ Pipeline technique

### 1. Génération de données — Modèle factoriel 3 facteurs

Les données synthétiques reproduisent les propriétés statistiques d'actions européennes :

```
R_i,t = β_marché,i × F_marché,t + β_sectoriel,i × F_secteur,t + ε_i,t
```

- **Facteur marché** : μ = 8% annualisé, σ = 15% (drift + diffusion brownienne)
- **Facteur sectoriel** : 3 secteurs distincts, σ = 8%
- **Terme idiosyncratique** : σ ∈ [10%, 20%], propre à chaque titre

### 2. Sélection du panier — Algorithme glouton

Sélection itérative des titres qui maximisent la corrélation avec **l'erreur résiduelle courante** (projection orthogonale à chaque étape) :

```python
résiduel_0 = R_indice
pour k = 1..N:
    i* = argmax_i  corr(R_i, résiduel_{k-1})
    résiduel_k = résiduel_{k-1} - β_i* × R_i*
```

### 3. Optimisation des poids — SLSQP (SciPy)

```
min_w   TE²(w) = Var(R_panier @ w - R_indice) × 252
s.t.    Σ w_i = 1         (contrainte budget)
        w_i ≥ 0           (long-only)
        w_i ≤ w_max       (concentration max, défaut 30%)
```

Résolution par **Sequential Least-Squares Programming** avec gradient analytique :

```
∂(TE²)/∂w = (2T/N) × X^T × (Xw - r_idx - μ_diff)
```

### 4. Backtesting walk-forward

Évaluation **out-of-sample sans look-ahead bias** :
- Fenêtre d'entraînement roulante : 504 jours (2 ans)
- Rebalancement trimestriel (63 jours)
- Ré-optimisation du panier à chaque rebalancement

---

## 📈 Résultats

| Taille panier | TE annualisée | Information Ratio | Max Drawdown | UCITS |
|:---:|:---:|:---:|:---:|:---:|
| 5 titres  | 5.20% | +0.12 | -19.5% | ❌ |
| 10 titres | 2.92% | -0.17 | -15.3% | ❌ |
| 15 titres | 2.01% | -0.10 | -13.5% | ❌ |
| **20 titres** | **1.37%** | **-0.38** | **-13.1%** | **✅** |

> ⚡ **Conclusion** : Le seuil réglementaire UCITS (TE < 1.5%) est franchi entre 15 et 20 titres, illustrant le compromis **précision de réplication / coût de transaction** central en gestion passive.

![Dashboard](output/dashboard.png)

---

## 🚀 Installation & Lancement

### Prérequis
- Python 3.10+

### Local (script CLI)

```bash
# Cloner le repo
git clone https://github.com/TON_USERNAME/index-replication-optimizer.git
cd index-replication-optimizer

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer
python index_replication.py
```

### App interactive (Streamlit)

```bash
streamlit run app.py
```

Ouvre automatiquement `http://localhost:8501`

### App en ligne

👉 **[Accéder à la démo live](https://TON_USERNAME-index-replication.streamlit.app)**

---

## 🛠️ Stack technique

| Librairie | Usage |
|---|---|
| `NumPy` | Calcul vectoriel, optimisation numérique |
| `Pandas` | Manipulation des séries temporelles |
| `SciPy` | Solveur SLSQP (optimisation sous contraintes) |
| `Matplotlib` | Visualisation (dashboard 4 panneaux) |
| `Streamlit` | Interface web interactive |

---

## 📚 Concepts financiers abordés

- **Tracking Error** (ex-ante vs ex-post)
- **Réplication physique partielle** vs réplication complète
- **Optimisation quadratique** sous contraintes de portefeuille
- **Walk-forward backtesting** et gestion du look-ahead bias
- **Information Ratio**, VaR/CVaR historique, Maximum Drawdown
- **Contraintes réglementaires UCITS** pour ETFs indiciels
- **Rebalancement dynamique** et impact sur les coûts de transaction

---

## 👤 Auteur

**Mohamed Amine El Aouadi**  
MSc Big Data & Finance Quantitative — NEOMA Business School (2026–2027)  
Ancien Analyste RFP — BNP Paribas Asset Management (2022–2026)  
📧 elaouadiamine02@gmail.com

---

*Projet réalisé dans le cadre de la candidature à l'alternance Front Office Risk Officer ETF/Index, BNP Paribas Asset Management.*
