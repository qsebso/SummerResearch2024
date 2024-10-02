import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the CSV data
file = 'analysis_output_REBEL.csv'
data = pd.read_csv(file)

# Load the JSON file with text data
with open('REBEL_outputs.json', 'r') as f:
    rebel_data = json.load(f)

# Extract texts and ensure they align with the CSV data
# Assuming the JSON file has a list of documents with 'text' and some identifier like 'doc_id' or 'index'
texts = [doc['text'] for doc in rebel_data]

# Define features and target variable
features = [
    'article_length', 'num_sentences', 'avg_word_length',
    'unique_entries', 'num_relations', 'relation_density', 'relation_density_per_sentence'
]
target = 'f1'

# Split the data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(X, y, texts, test_size=0.2, random_state=42)

# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Get feature importance (coefficients for linear regression)
coefficients = model.coef_
feature_importance = pd.DataFrame({'feature': features, 'importance': coefficients})
print(feature_importance.sort_values(by='importance', ascending=False))

# Provide suggestions based on feature importance
def suggest_improvements(text_data):
    suggestions = []
    
    pass

# Example text to analyze and suggest improvements
sample_index = 0  # Example: using the first text from the test set
sample_text = X_test.iloc[sample_index]
sample_actual_text = text_test[sample_index]

# Display the actual text and sample text features
print("Actual Text:")
print(sample_actual_text)

print("\nSample text features:")
print(sample_text)

# Print suggestions for improvement
print("\nSuggestions for improvement:")
print(suggest_improvements(sample_text))
