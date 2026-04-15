import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'transaction_amount': np.random.uniform(10, 5000, n_samples),
        'otp_request_frequency': np.random.randint(1, 10, n_samples),
        'device_info': np.random.choice(['Mobile', 'Laptop', 'Tablet'], n_samples),
        'location': np.random.choice(['Mumbai', 'Jaipur', 'Delhi', 'Bangalore'], n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]) # 10% Fraud scenarios [cite: 85]
    }
    return pd.DataFrame(data)
def preprocess_data(df):

    df = df.drop_duplicates().fillna(df.median(numeric_only=True))

    le_device = LabelEncoder()
    le_location = LabelEncoder()
    df['device_info'] = le_device.fit_transform(df['device_info'])
    df['location'] = le_location.fit_transform(df['location'])
     
    scaler = StandardScaler()
    df[['transaction_amount', 'otp_request_frequency']] = scaler.fit_transform(
        df[['transaction_amount', 'otp_request_frequency']]
    )
    return df

def run_fraud_system():
    
    df = generate_sample_data()

    df = preprocess_data(df)
    
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
  
    predictions = model.predict(X_test)
    print(f"--- MODEL PERFORMANCE ---")
    print(f"Final Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    

    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Purples')
    plt.title('Confusion Matrix: Fake OTP Scam Detection')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

run_fraud_system()