import numpy as np
import matplotlib.pyplot as plt

def visualize_attention_map(selected_cols, attention_scores, output_file):
    """
    Create and save a visualization of an attention map.

    Args:
    - selected_cols (numpy.ndarray): An array of shape (num_rows, num_selected_cols) containing the selected column indices for each row.
    - your_attention_scores (numpy.ndarray): An array of shape (num_rows, num_selected_cols) containing the attention scores for the selected columns.
    - output_file (str): The path to save the output image file.
    """
    # Create an empty 1024x1024 matrix filled with -128 (np.int8)
    n=selected_cols.shape[0]
    attention_map = np.full((n, n), -128, dtype=np.int8)

    # Fill the attention_map with the given attention scores
    rows = np.arange(n).reshape(-1, 1)
    attention_map[rows, selected_cols] = attention_scores.reshape(-1, selected_cols.shape[1]).astype(np.int8)

    # Visualize and save the attention map as an image
    plt.figure(figsize=(10, 10))
    plt.imshow(attention_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Attention Map')
    plt.xlabel('Output Sequence')
    plt.ylabel('Input Sequence')
    plt.savefig(output_file)
    plt.close()

# Example usage
# selected_cols = np.random.choice(1024, size=(1024, 128), replace=False)
selected_cols = np.random.choice(1024, size=(1024, 128))
your_attention_scores = np.random.rand(1024, 128)  # Example attention scores
output_file = 'attention_map.png'

visualize_attention_map(selected_cols, your_attention_scores, output_file)