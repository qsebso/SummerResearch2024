import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data from the CSV file
file_first = 'analysis_output.csv'
file = 'analysis_output_REBEL.csv'
df = pd.read_csv(file)

# Define the features we want to plot against precision, recall, and F1 scores
features = [
    'article_length', 'num_sentences', 'avg_word_length',
    'unique_entries', 'num_relations', 'relation_density', 'relation_density_per_sentence'
]

# Create a figure with subplots for Precision, Recall, and F1 Score
fig, axs = plt.subplots(len(features), 3, figsize=(12, 15))
fig.suptitle(file + ' Feature Correlation with Precision, Recall, and F1 Score', fontsize=16)

for i, feature in enumerate(features):
    # Plotting feature vs. Precision
    axs[i, 0].scatter(df[feature], df['precision'], alpha=0.3, s=10)
    axs[i, 0].set_title(f'{feature} vs. Precision')
    axs[i, 0].set_xlabel(feature)
    axs[i, 0].set_ylabel('Precision')
    axs[i, 0].grid(True)

    # Plotting feature vs. Recall
    axs[i, 1].scatter(df[feature], df['recall'], alpha=0.3, s=10)
    axs[i, 1].set_title(f'{feature} vs. Recall')
    axs[i, 1].set_xlabel(feature)
    axs[i, 1].set_ylabel('Recall')
    axs[i, 1].grid(True)

    # Plotting feature vs. F1 Score
    axs[i, 2].scatter(df[feature], df['f1'], alpha=0.3, s=10)
    axs[i, 2].set_title(f'{feature} vs. F1 Score')
    axs[i, 2].set_xlabel(feature)
    axs[i, 2].set_ylabel('F1 Score')
    axs[i, 2].set_ylim(0, 1)  # Assuming F1 scores are between 0 and 1
    axs[i, 2].grid(True)

# Adjust layout and show the plots
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure as an image file with the same name as the CSV file but with a .png extension
output_filename = os.path.splitext(file)[0] + '_graphs.png'
plt.savefig(output_filename)

plt.show()
