# House Price Prediction - My First ML Project
# By Srihitha

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create our own dataset
print("Creating dataset...")
np.random.seed(42)
n = 500

size = np.random.randint(500, 4000, n)
bedrooms = np.random.randint(1, 6, n)
age = np.random.randint(1, 50, n)
distance = np.random.randint(1, 30, n)

price = (size * 150) + (bedrooms * 20000) - (age * 500) - (distance * 1000) + np.random.randint(-10000, 10000, n)

df = pd.DataFrame({
    'Size_sqft': size,
    'Bedrooms': bedrooms,
    'Age_years': age,
    'Distance_km': distance,
    'Price': price
})

print("Dataset created successfully!")
print(df.head())
print("\nShape:", df.shape)

# Step 2: Prepare data
X = df.drop('Price', axis=1)
y = df['Price']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Step 4: Train model
print("\nTraining the model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully!")

# Step 5: Test model
y_pred = model.predict(X_test)

# Step 6: Results
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n--- Model Results ---")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score (Accuracy): {r2:.2f}")
print("(R2 Score closer to 1.0 means better accuracy)")

# Step 7: Visualize
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("results.png")
print("\nChart saved as results.png!")
print("\nProject Complete! Great job Srihitha! 🎉")