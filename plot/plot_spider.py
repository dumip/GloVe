import numpy as np
import matplotlib.pyplot as plt
from math import pi
from common import find_distiguishing_embeddings_subset

# Assuming 'words_of_interest' and 'embeddings_subset' are defined as before
words_of_interest = ['king', 'queen', 'prince', 'princess', 'man', 'woman', 'boy', 'girl']
embeddings_subset = find_distiguishing_embeddings_subset(words_of_interest)

# Define the number of variables and their names
num_vars = 3
labels = np.array(['Royalty', 'Adulthood', 'Gender'])

# Compute angle each bar is centered on:
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The radar chart requires a closed loop:
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], labels)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=7)
plt.ylim(0,1)

# Plot data and fill with color for each word
for i, word in enumerate(words_of_interest):
    data = embeddings_subset[i].tolist()
    data += data[:1]  # Complete the loop
    ax.plot(angles, data, linewidth=1, linestyle='solid', label=word)
    ax.fill(angles, data, alpha=0.1)

# Add a legend
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

plt.show()
