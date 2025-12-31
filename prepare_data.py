"""
Download 'Give Me Some Credit' dataset from Kaggle
Dataset: https://www.kaggle.com/c/GiveMeSomeCredit/data

For now, we'll use a similar synthetic dataset that mimics real credit data structure
In production, replace with actual Kaggle download
"""
import pandas as pd
import numpy as np

def create_realistic_credit_data():
    """Generate realistic credit data based on Give Me Some Credit format"""
    np.random.seed(42)
    n = 50000  # Larger dataset
    
    # Realistic distributions based on actual credit data
    data = {
        'RevolvingUtilizationOfUnsecuredLines': np.clip(np.random.beta(2, 5, n), 0, 1.5),
        'age': np.random.normal(52, 14, n).clip(18, 100).astype(int),
        'NumberOfTime30-59DaysPastDueNotWorse': np.random.poisson(0.3, n).clip(0, 20),
        'DebtRatio': np.random.exponential(0.3, n).clip(0, 5),
        'MonthlyIncome': np.random.lognormal(10.2, 0.8, n).clip(0, 500000),
        'NumberOfOpenCreditLinesAndLoans': np.random.poisson(8, n).clip(0, 60),
        'NumberOfTimes90DaysLate': np.random.poisson(0.2, n).clip(0, 20),
        'NumberRealEstateLoansOrLines': np.random.poisson(1, n).clip(0, 20),
        'NumberOfTime60-89DaysPastDueNotWorse': np.random.poisson(0.2, n).clip(0, 20),
        'NumberOfDependents': np.random.poisson(0.8, n).clip(0, 10)
    }
    
    df = pd.DataFrame(data)
    
    # Create target with realistic default rate (~6-7%)
    default_prob = (
        0.15 * df['RevolvingUtilizationOfUnsecuredLines'] +
        0.25 * (df['NumberOfTime30-59DaysPastDueNotWorse'] / 5) +
        0.30 * (df['NumberOfTimes90DaysLate'] / 3) +
        0.10 * (df['DebtRatio'] / 2) +
        0.05 * (1 - df['age'] / 100) +
        np.random.normal(0, 0.05, n)
    )
    
    df['SeriousDlqin2yrs'] = (default_prob > np.percentile(default_prob, 93)).astype(int)
    
    # Handle missing values realistically
    missing_income_idx = np.random.choice(n, int(n * 0.05), replace=False)
    df.loc[missing_income_idx, 'MonthlyIncome'] = np.nan
    
    return df

if __name__ == '__main__':
    df = create_realistic_credit_data()
    df.to_csv('credit_data.csv', index=False)
    print(f"Created dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Default rate: {df['SeriousDlqin2yrs'].mean()*100:.2f}%")
