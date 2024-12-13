import matplotlib.pyplot as plt
import json
import os

# Load the data from the JSON file
prefix = "/home/tangzichen/OpenFedLLM/output/ShenNong_TCM_Dataset_fedavg_client4_sample4_20241209234224"
file_path = os.path.join(prefix, "client_train_loss.json")
with open(file_path, 'r') as file:
    data = json.load(file)

# Convert string keys to integers if necessary
data = {int(k): v for k, v in data.items()}

# Determine the number of subplots needed based on the number of clients
num_clients = len(data)
cols = 2  # Set the number of columns to a fixed value
rows = (num_clients + cols - 1) // cols  # Calculate rows needed based on the number of clients

# Setup the figure for plotting
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Adjust the size dynamically based on the number of rows
axes = axes.flatten()

# Process each client's data
for i, (client_id, losses) in enumerate(data.items()):
    for epoch, loss_values in enumerate(losses):
        axes[i].plot(loss_values, label=f'Epoch {epoch + 1}')
    axes[i].set_title(f'Client {client_id} Training Loss')
    axes[i].set_xlabel('Local Iterations')
    axes[i].set_ylabel('Training Loss')
    axes[i].legend()

# Deactivate unused axes if any
if num_clients % cols != 0:
    for j in range(num_clients, rows * cols):
        axes[j].axis('off')

# Proper use of tight_layout to adjust the layout
fig.tight_layout()

# Save the figure as a PDF
pdf_path = os.path.join(prefix, 'training_loss_plots.pdf')
fig.savefig(pdf_path)

plt.show()
print(f'Training loss plots saved to {pdf_path}')
