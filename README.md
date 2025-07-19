# 🧠aave-credit-scoring DeFi Credit Scoring Engine (Aave V2)

This project builds a robust, interpretable credit scoring system for DeFi wallets using real Aave V2 transaction data.  
We apply domain-specific heuristics and a LightGBM model to score wallets from **0 to 1000**.  
The result is a risk-aware snapshot of any wallet’s on-chain reputation.

---

## 📁 File Structure

```
📦aave-credit-scoring/
├── credit_score_generator.py   # Main pipeline to generate features + scores
├── user_transactions.json      # Input: 100K transaction records
├── wallet_scores.csv           # Output: Wallet addresses with credit scores
├── score_distribution.png      # Visual: Histogram of credit scores
├── score_factors.png           # Visual: SHAP summary of score drivers
├── credit_model.pkl            # Trained LightGBM model
├── scaler.pkl                  # Normalization scaler
├── analysis.md                 # Exploratory metrics + EDA
├── README.md                   # You're reading it now
└── .gitignore                  # Cache, models, images, csv files
```

---

## 📊 Real Run Summary

```bash
📂 Loading transaction data...
✔ Loaded 100,000 transactions from 3,497 wallets
🛠️ Calculating wallet features...
🧮 Generating credit scores...
📊 Creating explainability visuals...

✅ Successfully scored 3497 wallets
📋 Score distribution summary:
count    3497.000000
mean      507.035745
std       122.609934
min       318.000000
25%       399.000000
50%       432.000000
75%       616.000000
max       706.000000
```

---

### 🔝 Top 5 Wallets

| Wallet Address | Credit Score |
|----------------|--------------|
| `0x0168...cacb` | **706** |
| `0x05e5...3343` | **705** |
| `0x0269...e2f2d` | **705** |
| `0x060c...8212` | **703** |
| `0x02e9...076d` | **703** |

---

### 🔻 Bottom 5 Wallets

| Wallet Address | Credit Score |
|----------------|--------------|
| `0x0531...9bd6` | **318** |
| `0x04dd...36bf` | **327** |
| `0x035f...1c29` | **346** |
| `0x0611...5f65` | **348** |
| `0x0091...788a` | **349** |

---

## 💡 Score Interpretation

```python
SCORE_BANDS = {
    (800, 1000): "Excellent (Top-tier behavior)",
    (600, 799):  "Good (Reliable users)",
    (400, 599):  "Average (Some risk factors)",
    (200, 399):  "Risky (Irregular or volatile)",
    (0, 199):    "Critical (Likely bots/liquidated)"
}
```

---

## 🧠 Feature Highlights

Wallets are scored based on their:

- 🏦 **Repayment behavior** (repay-to-borrow ratio)
- ⚖️ **Risk signals** (liquidations, inactivity)
- ⏳ **Temporal patterns** (tx frequency, 24/7 bot score)
- 💰 **Financial activity** (total and average tx size)
- 🧍 **Human indicators** (time of day, activeness)

---

## 🖼️ Visuals (Click to Open)

- 📊 [Score Distribution Histogram](./score_distribution.png)
- 🧠 [SHAP Feature Importance Plot](./score_factors.png)

---

## 🚀 Quick Start

```bash
# Install required dependencies
pip install pandas numpy scikit-learn lightgbm shap matplotlib seaborn joblib

# Run scoring
python credit_score_generator.py
```

To retrain model with new data:
```bash
python credit_score_generator.py --retrain new_transactions.json
```

---

## 🛠️ Uses & Applications

- 🧾 On-chain credit reputation system
- 💳 Pre-screening for DeFi lending
- 🔍 Research on wallet behavior clusters
- 🧪 Risk analysis / anomaly detection

---

## 📜 License

MIT — free to use commercially or academically

---


