import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from common import load_glove_embeddings, find_distinguishing_feature, find_distiguishing_embeddings_subset

# Define the words of interest
words_of_interest = ['king', 'queen', 'prince', 'princess', 'man', 'woman', 'boy', 'girl']

embeddings_subset = find_distiguishing_embeddings_subset(words_of_interest)

# Create the heatmap for the selected features
plt.figure(figsize=(10, 6))
sns.heatmap(embeddings_subset, annot=True, fmt=".4f", cmap='coolwarm', 
            xticklabels=['Royalty?', 'Gender?', 'Adulthood?'], yticklabels=words_of_interest)

plt.title('Heatmap of Selected Features for Selected Words')
plt.xlabel('Feature')
plt.ylabel('Word')
plt.show()

