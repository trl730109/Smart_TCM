import matplotlib.pyplot as plt
import json

# Load the data from the JSON file
file_path = "/home/tangzichen/OpenFedLLM/output/ShenNong_TCM_Dataset_fedavg_client8_sample8_20241107092655/client_train_loss.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# Setup the figure for plotting
plt.figure(figsize=(15, 10))  # You can adjust the size as necessary

# Process each client's data
for client_id, losses in data.items():
    # Create a subplot for each client
    plt.subplot(2, 2, int(client_id) + 1)  # Adjust based on the number of clients
    for epoch, loss_values in enumerate(losses):
        # Plot each epoch's loss values
        plt.plot(loss_values, label=f'Epoch {epoch + 1}')
    
    plt.title(f'Client {client_id} Training Loss')
    plt.xlabel('Local Iterations')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.tight_layout()

# Save the figure as a PDF
plt.savefig('training_loss_plots.pdf')

plt.show()
