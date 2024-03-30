# plot_heatmap.py
import matplotlib.pyplot as plt
import seaborn as sns
from load_embeddings import load_glove_embeddings  # Import the function from the other file

# Define the words for which you want to load embeddings
words_of_interest = ['king', 'queen', 'prince', 'princess', 'man', 'woman', 'boy', 'girl']

# Load GloVe embeddings for the specified words
words, embeddings = load_glove_embeddings('../vectors.txt', words_of_interest)

# Select only the first N features to plot for simplicity and clarity
N = 5  # For example, use the first 5 features
embeddings_subset = embeddings[:, :N]

# Create the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(embeddings_subset, annot=True, fmt=".2f", cmap='viridis', xticklabels=range(1, N+1), yticklabels=words)

plt.title('Heatmap of the first 5 features for selected words')
plt.xlabel('Feature')
plt.ylabel('Word')
plt.show()
