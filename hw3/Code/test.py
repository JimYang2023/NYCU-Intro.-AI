# test
import pandas as pd

# Define mapping from numbers to animal names
label_map = {0: "elephant", 1: "jaguar", 2: "lion", 3: "parrot", 4: "penguin"}

# Load CSV files
df_truth = pd.read_csv("test_labels.csv")
df_pred = pd.read_csv("CNN.csv")           # contains prediction numbers like 0–4

# Drop headers/columns you don't need
true_labels = df_truth['Animal']
predicted_labels = df_pred['prediction'].map(label_map)  # convert number to string

min_len = min(len(true_labels), len(predicted_labels))
true_labels = true_labels[:min_len]
predicted_labels = predicted_labels[:min_len]

matches = (true_labels == predicted_labels)
num_correct = matches.sum()
total = len(matches)
accuracy = num_correct / total * 100

print(f"CNN Accuracy: {accuracy:.2f}%")

# Decision Tree
df_pred = pd.read_csv("DecisionTree.csv")

predicted_labels = df_pred['prediction'].map(label_map)
matches = (true_labels == predicted_labels)
num_correct = matches.sum()
total = len(matches)
accuracy = num_correct / total * 100

print(f"DecisionTree Accuracy: {accuracy:.2f}%")