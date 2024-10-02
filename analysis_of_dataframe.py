import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data from the CSV file
file = 'analysis_output_REBEL.csv'
data = pd.read_csv(file)

# Ensure the output directory exists
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Compute the correlation matrix
correlation_scores = data.corr()
print(correlation_scores)

# Plot the full correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_scores, annot=True, cmap='coolwarm', center=0)
plt.title('Full Correlation Matrix')
plt.savefig(os.path.join(output_dir, 'full_correlation_matrix.png'))
plt.show()

# Extract correlations of features with performance metrics
performance_metrics = ['precision', 'recall', 'f1']
relevant_correlations = correlation_scores[performance_metrics].loc[data.columns[:-3]]  # Exclude the last three columns which are the performance metrics

# Plot the relevant correlations
plt.figure(figsize=(10, 8))
sns.heatmap(relevant_correlations, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation with Performance Metrics')
plt.savefig(os.path.join(output_dir, 'feature_performance_correlations.png'))
plt.show()

# Define the features to plot
features = [
    'article_length', 'num_sentences', 'avg_word_length',
    'unique_entries', 'num_relations', 'relation_density', 'relation_density_per_sentence'
]

# Create scatter plots for each feature against precision, recall, and F1 score
fig, axs = plt.subplots(len(features), 3, figsize=(15, 20))
fig.suptitle('Feature Correlation with Precision, Recall, and F1 Score', fontsize=16)

for i, feature in enumerate(features):
    axs[i, 0].scatter(data[feature], data['precision'], alpha=0.3, s=10)
    axs[i, 0].set_title(f'{feature} vs. Precision')
    axs[i, 0].set_xlabel(feature)
    axs[i, 0].set_ylabel('Precision')

    axs[i, 1].scatter(data[feature], data['recall'], alpha=0.3, s=10)
    axs[i, 1].set_title(f'{feature} vs. Recall')
    axs[i, 1].set_xlabel(feature)
    axs[i, 1].set_ylabel('Recall')

    axs[i, 2].scatter(data[feature], data['f1'], alpha=0.3, s=10)
    axs[i, 2].set_title(f'{feature} vs. F1 Score')
    axs[i, 2].set_xlabel(feature)
    axs[i, 2].set_ylabel('F1 Score')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(output_dir, 'feature_correlation_plots.png'))
plt.show()

# Plot density distributions for each feature
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.savefig(os.path.join(output_dir, f'distribution_{feature}.png'))
    plt.show()
