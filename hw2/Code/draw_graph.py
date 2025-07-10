import matplotlib.pyplot as plt

# Data
depths = [1, 2, 3, 4]
minimax_times = [5734.41, 48492.33, 620261.42, 1695589.38]
alpha_beta_times = [5104.48, 33768.19, 393416.11, 748913.98]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(depths, minimax_times, marker='o', linestyle='-', color='r', label="Minimax")
plt.plot(depths, alpha_beta_times, marker='s', linestyle='--', color='b', label="Alpha-Beta")

# Labels and title
plt.xlabel("Search Depth")
plt.ylabel("Execution Time")
plt.title("Execution Time vs Search Depth")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()