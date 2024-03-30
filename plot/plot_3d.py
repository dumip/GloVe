import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from common import load_glove_embeddings, find_distinguishing_feature, find_distiguishing_embeddings_subset

# Define the words of interest
words_of_interest = ['king', 'queen', 'prince', 'princess', 'man', 'woman', 'boy', 'girl']

embeddings_subset = find_distiguishing_embeddings_subset(words_of_interest)

# Initialize a 3D subplot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# X, Y, Z coordinates for each word of interest
X = embeddings_subset[:, 0]  # Feature representing Royalty
Y = embeddings_subset[:, 1]  # Feature representing Gender
Z = embeddings_subset[:, 2]  # Feature representing Adulthood

# Scatter plot for each word
for i, word in enumerate(words_of_interest):
    ax.scatter(X[i], Y[i], Z[i], label=word)

ax.set_xlabel('Royalty Feature')
ax.set_ylabel('Gender Feature')
ax.set_zlabel('Adulthood Feature')
ax.set_title('3D Plot of Selected Features for Selected Words')
ax.legend()

plt.show()
