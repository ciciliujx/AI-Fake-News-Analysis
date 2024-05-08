import numpy as np
import matplotlib.pyplot as plt

# Categories
categories = ['X', 'Y']

# Data
group_A = [5, 10]
group_B = [3, 12]
group_C = [2, 8]

# Position of bars on the x-axis
ind = np.arange(len(categories))

# Size of the plot
plt.figure(figsize=(8, 6))

# Plotting
plt.bar(ind, group_A, label='Group A')
plt.bar(ind, group_B, bottom=group_A, label='Group B')
# For Group C, the bottom is the sum of Group A and Group B
plt.bar(ind, group_C, bottom=np.add(group_A, group_B), label='Group C')

# Labels, Title, and Legend
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Stacked Bar Chart')
plt.xticks(ind, categories)
plt.legend()

# Show the plot
plt.show()
