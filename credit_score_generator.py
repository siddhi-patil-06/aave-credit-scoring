import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings

# Configuration
CONFIG = {
    'score_range': (0, 1000),
    'risk_factors': {
        'liquidation_penalty': -200,
        'high_frequency_penalty': -150,
        'repayment_reward': 300,
        'longevity_reward': 2,  # per day
        'deposit_bonus': 50,
        'borrow_penalty': -30,
        'bot_penalty': -100  # For unusual time patterns
    }
}

def load_data(filepath):
    """Load and preprocess transaction data"""
    print("📂 Loading transaction data...")
    with open(filepath) as f:
        data = json.load(f)
    
    # Normalize the nested structure
    df = pd.json_normalize(data, sep='_')
    
    # Convert and rename fields
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['value'] = pd.to_numeric(df['actionData_amount'], errors='coerce').fillna(0)
    df['action'] = df['action'].str.lower()
    df['asset'] = df['actionData_assetSymbol'].str.lower()
    df.rename(columns={'userWallet': 'user'}, inplace=True)
    
    print(f"✔ Loaded {len(df):,} transactions from {df['user'].nunique():,} wallets")
    return df

def calculate_wallet_features(wallet_txs):
    """Calculate comprehensive features for a single wallet"""
    # Time-based features
    timestamps = wallet_txs['timestamp'].astype('int64') // 10**9
    time_diffs = np.diff(np.sort(timestamps)) if len(timestamps) > 1 else np.array([0])
    hours = wallet_txs['timestamp'].dt.hour
    
    # Action counts
    action_counts = wallet_txs['action'].value_counts(normalize=True).to_dict()
    
    features = {
        'tx_count': len(wallet_txs),
        'unique_assets': wallet_txs['asset'].nunique(),
        'days_active': (timestamps.max() - timestamps.min()) / 86400 if len(timestamps) > 1 else 0,
        'tx_frequency_std': np.std(time_diffs) if len(time_diffs) > 1 else 0,
        'deposit_ratio': action_counts.get('deposit', 0),
        'borrow_ratio': action_counts.get('borrow', 0),
        'repay_ratio': action_counts.get('repay', 0),
        'redeem_ratio': action_counts.get('redeemunderlying', 0),
        'liquidation_count': action_counts.get('liquidationcall', 0),
        'avg_tx_value': wallet_txs['value'].mean(),
        'value_std': wallet_txs['value'].std(),
        'total_value': wallet_txs['value'].sum(),
        'night_tx_ratio': ((hours < 6) | (hours > 22)).mean(),
        'workhour_tx_ratio': ((hours >= 9) & (hours <= 17)).mean(),
        'bot_likelihood': 1 if (np.mean(time_diffs) < 60 and len(time_diffs) > 10) else 0  # ← fixed here
    }
    return features

def generate_scores(wallet_features):
    """Generate final credit scores"""
    feature_cols = [col for col in wallet_features.columns 
                   if col not in ['user', 'base_score', 'credit_score']]
    X = wallet_features[feature_cols]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(X)

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    model.fit(scaled_features, wallet_features['base_score'])

    scores = model.predict(scaled_features)
    scores = np.clip(scores, *CONFIG['score_range'])

    return scores, model, scaler

def main():
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    df = load_data("user_transactions.json")
    
    print("🛠️ Calculating wallet features...")
    features_list = []
    for wallet, group in df.groupby('user'):
        features = calculate_wallet_features(group)
        features['user'] = wallet
        features_list.append(features)
    
    wallet_features = pd.DataFrame(features_list)
    
    risk = CONFIG['risk_factors']
    wallet_features['base_score'] = (
        500 +
        risk['repayment_reward'] * wallet_features['repay_ratio'] +
        risk['longevity_reward'] * wallet_features['days_active'] +
        risk['deposit_bonus'] * wallet_features['deposit_ratio'] +
        risk['borrow_penalty'] * wallet_features['borrow_ratio'] +
        risk['liquidation_penalty'] * wallet_features['liquidation_count'] +
        risk['high_frequency_penalty'] * (wallet_features['tx_frequency_std'] < 60).astype(int) +
        risk['bot_penalty'] * wallet_features['bot_likelihood']
    ).clip(300, 700)
    
    print("🧮 Generating credit scores...")
    scores, model, scaler = generate_scores(wallet_features)
    wallet_features['credit_score'] = scores.astype(int)
    
    wallet_features[['user', 'credit_score']].to_csv('wallet_scores.csv', index=False)
    joblib.dump(model, 'credit_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("📊 Creating explainability visuals...")
    explainer = shap.Explainer(model)
    feature_cols = [col for col in wallet_features.columns 
                   if col not in ['user', 'base_score', 'credit_score']]
    shap_values = explainer(scaler.transform(wallet_features[feature_cols]))
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, wallet_features[feature_cols], show=False)
    plt.tight_layout()
    plt.savefig('score_factors.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(wallet_features['credit_score'], bins=20, kde=True)
    plt.title('Credit Score Distribution', fontsize=14)
    plt.xlabel('Credit Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig('score_distribution.png', dpi=300)
    plt.close()
    
    print("\n✅ Successfully scored {} wallets".format(len(wallet_features)))
    print("📋 Score distribution summary:")
    print(wallet_features['credit_score'].describe())
    
    print("\n💡 Interpretation Guide:")
    print("800-1000: Excellent (responsible, consistent repayments)")
    print("600-799: Good (reliable users)")
    print("400-599: Average (some risk factors)")
    print("200-399: Risky (irregular patterns)")
    print("0-199: High risk (liquidations/bot-like)")
    
    print("\n🔝 Top 5 Wallets:")
    print(wallet_features[['user', 'credit_score']].sort_values('credit_score', ascending=False).head().to_string(index=False))
    
    print("\n🔻 Bottom 5 Wallets:")
    print(wallet_features[['user', 'credit_score']].sort_values('credit_score').head().to_string(index=False))

if __name__ == "__main__":
    main()
