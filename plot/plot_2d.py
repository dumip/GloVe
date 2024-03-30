import matplotlib.pyplot as plt
from common import find_distiguishing_embeddings_subset

# Assuming 'words_of_interest' and 'find_distiguishing_embeddings_subset' are defined as before
words_of_interest = ['king', 'queen', 'prince', 'princess', 'man', 'woman', 'boy', 'girl']
embeddings_subset = find_distiguishing_embeddings_subset(words_of_interest)

# Use only the first two features for a 2D plot
X = embeddings_subset[:, 0]  # Feature 1 (e.g., Royalty)
Y = embeddings_subset[:, 1]  # Feature 2 (e.g., Gender)

plt.figure(figsize=(10, 6))
for i, word in enumerate(words_of_interest):
    plt.scatter(X[i], Y[i], marker='o')
    plt.text(X[i], Y[i], word, fontsize=9)

plt.xlabel('Feature 1 (Royalty?)')
plt.ylabel('Feature 2 (Gender?)')
plt.title('2D Scatter Plot of Word Embeddings')
plt.grid(True)
plt.show()
