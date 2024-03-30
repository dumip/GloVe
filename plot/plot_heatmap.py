import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from load_embeddings import load_glove_embeddings  # Assuming this is already defined

# Define the words of interest
words_of_interest = ['king', 'queen', 'prince', 'princess', 'man', 'woman', 'boy', 'girl']

def find_distinguishing_feature(high_words, low_words, embeddings, words):
    """
    Identify a single feature index where the first set of four words have the highest values,
    while the second set of four words have the lowest values.

    Parameters:
    - high_words: tuple. A tuple of four words expected to have high feature values.
    - low_words: tuple. A tuple of four words expected to have low feature values.
    - embeddings: np.ndarray. The array of word embeddings.
    - words: list. The list of words corresponding to the embeddings.

    Returns:
    - int. Index of the feature with the highest difference in mean values between the two sets.
    """
    if len(high_words) != 4 or len(low_words) != 4:
        raise ValueError("Both word tuples must contain exactly four words.")
    
    # Retrieve embeddings for both sets of words
    high_embeddings = np.array([embeddings[words.index(word)] for word in high_words])
    low_embeddings = np.array([embeddings[words.index(word)] for word in low_words])
    
    # Calculate the mean embedding for each set
    mean_high = np.mean(high_embeddings, axis=0)
    mean_low = np.mean(low_embeddings, axis=0)
    
    # Calculate the difference between the means
    diff = mean_high - mean_low
    
    # Identify the index of the feature with the highest value in the difference
    distinguishing_feature_index = np.argmax(diff)
    
    return distinguishing_feature_index

# Load GloVe embeddings for the specified words
words, embeddings = load_glove_embeddings('../vectors.txt', words_of_interest)

royalty_tuple_high = ('king', 'queen', 'prince', 'princess')
royalty_tuple_low = ('man', 'woman', 'boy', 'girl')
gender_tuple_high = ('king', 'man', 'boy', 'prince')
gender_tuple_low = ('queen', 'princess', 'woman', 'girl')
adulthood_tuple_high = ('man', 'woman', 'king', 'queen')
adulthood_tuple_low = ('prince', 'pricess', 'boy', 'girl')

# Find the most representative feature for each tuple
royalty_feature_index = find_distinguishing_feature(royalty_tuple_high, royalty_tuple_low, embeddings, words)
gender_feature_index = find_distinguishing_feature(gender_tuple_high, gender_tuple_low, embeddings, words)
adulthood_tuple = find_distinguishing_feature(adulthood_tuple_high, gender_tuple_low, embeddings, words)

print(f"Most representative feature for royalty: {royalty_feature_index}")
print(f"Most representative feature for gender: {gender_feature_index}")
print(f"Most representative feature for adulthood: {adulthood_tuple}")

feature_indexes = [royalty_feature_index, gender_feature_index, adulthood_tuple] 

# Retrieve the specific feature values for the words of interest
embeddings_subset = embeddings[[words.index(word) for word in words_of_interest]][:, feature_indexes]

# Create the heatmap for the selected features
plt.figure(figsize=(10, 6))
sns.heatmap(embeddings_subset, annot=True, fmt=".2f", cmap='coolwarm', 
            xticklabels=['Royalty', 'Gender', 'Adulthood'], yticklabels=words_of_interest)

plt.title('Heatmap of Selected Features for Selected Words')
plt.xlabel('Feature')
plt.ylabel('Word')
plt.show()

