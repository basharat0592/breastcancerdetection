import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

def load_and_clean_data():
    # Load dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Standardize features
    scaler = StandardScaler()
    features = df.drop('target', axis=1)
    df_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    df_scaled['target'] = df['target']

    # Save cleaned data
    df_scaled.to_csv('cleaned_data.csv', index=False)

if __name__ == '__main__':
    load_and_clean_data()
