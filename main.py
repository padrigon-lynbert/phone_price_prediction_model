import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import numpy as np
import matplotlib.pyplot as plt

# collection
dataset_file_path = 'D:\manualCDmanagement\codes\Projects\VMs\skl algorithms\Binary classification\\00_datasets\phone'
file_name = 'mobile_phone_price_prediction.csv'
dataset = os.path.join(dataset_file_path, file_name)
df = pd.read_csv(dataset)

# cleaning
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop('Rating', axis=1, inplace=True)

columns_to_drop = ['Rating', 'Spec_score', 'Price']
for column in df.columns:
    if column not in columns_to_drop:
        df[column], _ = pd.factorize(df[column])

df['Price'] = df['Price'].str.replace(',', '').astype(float)

# splitting

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# model creation

lasso_model = Lasso(alpha=0.1)      
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)


x_line = np.linspace(min(y_test), max(y_test), 100)
y_line = x_line 

'''
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lasso, color='blue', alpha=0.2, label='Actual vs. Predicted')
plt.plot(x_line, y_line, color='k', linestyle='--', linewidth=2, label='Identity Line')
plt.title('Actual vs. Predicted Values - Lasso Regression')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()
'''

