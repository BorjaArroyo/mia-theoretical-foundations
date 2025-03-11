import json
import matplotlib.pyplot as plt

# Load results with error handling
try:
    with open("epsilon_results1.json", "r") as f:
        results = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading JSON file: {e}")
    exit(1)

# Ensure keys and values are properly sorted (if keys are numeric strings)
try:
    x = sorted(map(float, results.keys()))
    y = [results[str(k)] for k in x]
except ValueError as e:
    print(f"Error processing keys/values: {e}")
    exit(1)

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size as needed

# Plot data
ax.plot(x, y, marker='o', linestyle='-', color='b')

# Add labels and title
ax.set_xlabel(r"$\sigma$", fontsize=12)
ax.set_ylabel(r"$\epsilon$", fontsize=12)
# ax.set_title("Relationship between $\sigma$ and $\epsilon$", fontsize=14)

# Add grid (optional)
ax.grid(True, linestyle='--', alpha=0.7)

# Optimize layout and save
plt.tight_layout()
plt.savefig("epsilon_results1.png", dpi=300)  # High resolution for articles
plt.close(fig)  # Close figure to free memory