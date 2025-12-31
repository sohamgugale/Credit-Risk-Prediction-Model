"""
Load and prepare the real Kaggle 'Give Me Some Credit' dataset
Dataset: https://www.kaggle.com/c/GiveMeSomeCredit/data
"""
import pandas as pd
import numpy as np

def load_kaggle_data():
    """Load real Kaggle credit data"""
    
    # Load the real data
    df = pd.read_csv('cs-training.csv')
    
    # Remove the index column that Kaggle adds
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    print(f"\n📊 Real Kaggle Dataset Loaded:")
    print(f"   Total records: {len(df):,}")
    print(f"   Features: {len(df.columns)}")
    print(f"   Default rate: {df['SeriousDlqin2yrs'].mean()*100:.2f}%")
    print(f"\n📉 Missing Values:")
    print(df.isnull().sum())
    
    # Basic statistics
    print(f"\n📈 Target Distribution:")
    print(df['SeriousDlqin2yrs'].value_counts())
    
    return df

def clean_data(df):
    """Clean and prepare the data"""
    
    # Handle outliers in age
    df = df[df['age'] > 0]  # Remove invalid ages
    df = df[df['age'] < 100]  # Remove unrealistic ages
    
    # Handle extreme debt ratios (some are 50000+)
    df.loc[df['DebtRatio'] > 5, 'DebtRatio'] = 5
    
    # Handle missing MonthlyIncome (impute with median by age group)
    age_bins = [0, 25, 35, 50, 65, 100]
    df['AgeGroup'] = pd.cut(df['age'], bins=age_bins, labels=[1,2,3,4,5])
    
    for group in df['AgeGroup'].unique():
        median_income = df[df['AgeGroup'] == group]['MonthlyIncome'].median()
        df.loc[(df['AgeGroup'] == group) & (df['MonthlyIncome'].isnull()), 'MonthlyIncome'] = median_income
    
    # Fill remaining missing values
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
    df['NumberOfDependents'].fillna(0, inplace=True)
    
    # Feature engineering
    df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    df['TotalPastDue'] = (df['NumberOfTime30-59DaysPastDueNotWorse'] + 
                          df['NumberOfTime60-89DaysPastDueNotWorse'] + 
                          df['NumberOfTimes90DaysLate'])
    df['UtilizationToIncome'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['DebtRatio']
    
    # Drop temporary column
    df = df.drop('AgeGroup', axis=1)
    
    print(f"\n✅ Data cleaned:")
    print(f"   Final records: {len(df):,}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    return df

if __name__ == '__main__':
    # Load and clean
    df = load_kaggle_data()
    df_clean = clean_data(df)
    
    # Save cleaned data
    df_clean.to_csv('credit_data.csv', index=False)
    print(f"\n💾 Saved cleaned data to credit_data.csv")
    print(f"   Shape: {df_clean.shape}")
    print(f"   Default rate: {df_clean['SeriousDlqin2yrs'].mean()*100:.2f}%")
