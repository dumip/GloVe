import numpy as np

# Function to load GloVe embeddings for specific words
def load_glove_embeddings(file_path, words_of_interest):
    embeddings = []
    words = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            word = parts[0]
            if word in words_of_interest:
                vector = [float(part) for part in parts[1:]]  # Convert to float and keep the vector
                embeddings.append(vector)
                words.append(word)
                # Check if all words of interest are found
                if len(words) == len(words_of_interest):
                    break
    return words, np.array(embeddings)