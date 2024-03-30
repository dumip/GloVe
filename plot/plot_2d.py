import numpy as np
from load_embeddings import load_glove_embeddings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Define the words for which you want to load embeddings
words_of_interest = ['king', 'queen', 'prince', 'princess', 'man', 'woman', 'boy', 'girl']

# Load GloVe embeddings for the specified words
words, embeddings = load_glove_embeddings('../vectors.txt', words_of_interest)

# Adjust the perplexity parameter based on the number of samples
perplexity_value = min(30, len(words) - 1)  # Ensure perplexity is less than the number of samples

# Perform dimensionality reduction using t-SNE with adjusted perplexity
tsne = TSNE(n_components=2, perplexity=perplexity_value)

# Make sure to pass only 'embeddings' to the fit_transform method
reduced_embeddings = tsne.fit_transform(embeddings)

# Plot the reduced embeddings
plt.figure(figsize=(10, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

# Annotate the points with the corresponding words
for i, word in enumerate(words):
    plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

plt.show()