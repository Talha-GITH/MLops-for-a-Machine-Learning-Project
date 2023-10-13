# Project: Implementing Simplified MLops for a Machine Learning Project
# Branch: main (or the name of your current working branch)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load your dataset (you'll need to replace 'X' and 'y' with your actual dataset)
X, y = np.random.rand(100, 1), np.random.rand(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a machine learning model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Model accuracy: {accuracy}')
