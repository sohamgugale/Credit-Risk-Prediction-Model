"""
Advanced ML Pipeline with Hyperparameter Tuning and Cross-Validation
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, make_scorer
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess():
    """Load and preprocess data"""
    df = pd.read_csv('credit_data.csv')
    
    # Handle missing values
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
    df['NumberOfDependents'].fillna(0, inplace=True)
    
    # Feature engineering
    df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    df['TotalPastDue'] = (df['NumberOfTime30-59DaysPastDueNotWorse'] + 
                          df['NumberOfTime60-89DaysPastDueNotWorse'] + 
                          df['NumberOfTimes90DaysLate'])
    df['UtilizationToIncome'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['DebtRatio']
    df['AgeGroup'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], labels=[1,2,3,4,5])
    
    return df

def train_best_models(X_train, X_test, y_train, y_test):
    """Train models with hyperparameter tuning"""
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    models = {}
    
    # 1. Random Forest with tuning
    print("Training Random Forest...")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10],
        'class_weight': ['balanced']
    }
    rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, 
                      cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
    rf.fit(X_train_balanced, y_train_balanced)
    models['Random Forest'] = rf.best_estimator_
    print(f"Best params: {rf.best_params_}")
    
    # 2. XGBoost with tuning
    print("Training XGBoost...")
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7, 10],
        'learning_rate': [0.01, 0.1],
        'scale_pos_weight': [len(y_train_balanced[y_train_balanced==0]) / len(y_train_balanced[y_train_balanced==1])]
    }
    xgb = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), 
                       xgb_params, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
    xgb.fit(X_train_balanced, y_train_balanced)
    models['XGBoost'] = xgb.best_estimator_
    print(f"Best params: {xgb.best_params_}")
    
    # 3. Gradient Boosting
    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train_balanced, y_train_balanced)
    models['Gradient Boosting'] = gb
    
    # 4. Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train_balanced, y_train_balanced)
    models['Logistic Regression'] = lr
    
    return models

def calculate_business_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate business impact metrics"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Costs (example values)
    cost_of_default = 10000  # Average loss when loan defaults
    cost_of_rejection = 200   # Opportunity cost of rejecting good customer
    
    # Confusion matrix elements
    TP = ((y_pred == 1) & (y_true == 1)).sum()  # Correctly identified defaults
    FP = ((y_pred == 1) & (y_true == 0)).sum()  # False alarms
    TN = ((y_pred == 0) & (y_true == 0)).sum()  # Correctly approved
    FN = ((y_pred == 0) & (y_true == 1)).sum()  # Missed defaults
    
    # Business costs
    total_cost = (FN * cost_of_default) + (FP * cost_of_rejection)
    total_savings = TP * cost_of_default  # Prevented defaults
    net_value = total_savings - total_cost
    
    return {
        'Total Cost': total_cost,
        'Prevented Losses': total_savings,
        'Net Value': net_value,
        'Missed Defaults': FN,
        'False Rejections': FP
    }

if __name__ == '__main__':
    # Load data
    df = load_and_preprocess()
    
    X = df.drop('SeriousDlqin2yrs', axis=1)
    y = df['SeriousDlqin2yrs']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = train_best_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Evaluate and save
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\n{name} - ROC-AUC: {auc:.4f}")
        
        # Save best model
        if name == 'XGBoost':
            joblib.dump(model, 'best_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')
    
    print("\nModels saved successfully!")
