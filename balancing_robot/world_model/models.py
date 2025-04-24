import torch
import torch.nn as nn
import torch.optim as optim
import random

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.initial_hidden = nn.Parameter(torch.randn(num_layers, 1, hidden_size)) # Trainable initial hidden state

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = self.initial_hidden.repeat(1, batch_size, 1).to(x.device)  # Repeat for batch size
        out, _ = self.gru(x, h0)
        return out
        # out: (batch_size, seq_len, hidden_size)
        # return out[:, -1, :]  # Return the last output in the sequence


class Predictor(nn.Module):
    def __init__(self, embedding_size, action_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size + action_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, embedding, action):
        # embedding: (batch_size, embedding_size)
        # action: (batch_size, action_size)
        x = torch.cat([embedding, action], dim=-1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Hyperparameters
imu_size = 6
action_size = 2
input_size = imu_size + action_size  # imu_size + action_size
embedding_size = 32
predictor_hidden_size = 64
num_encoder_layers = 2
learning_rate = 0.001
num_epochs = 10
batch_size = 64
warmup_length = 10
num_predictions = 5
sequence_length = warmup_length + num_predictions # Total Sequence Length (Warmup + Prediction)
min_episode_length = 100

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the models
encoder = Encoder(input_size, embedding_size, num_encoder_layers).to(device)
predictor = Predictor(embedding_size, action_size, predictor_hidden_size).to(device) # Action size is always 2

# Define the optimizer
optimizer = optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=learning_rate)

# Define the loss function
loss_fn = nn.MSELoss()

# Generate some dummy data for training
def generate_dummy_data(num_episodes, episode_length_range, input_size):
    episodes = []
    for _ in range(num_episodes):
        episode_length = random.randint(episode_length_range[0], episode_length_range[1])
        episode = torch.randn(episode_length, input_size)
        episodes.append(episode)
    return episodes

# Example: Generate 10 episodes, each with a length between 100 and 200
episodes = generate_dummy_data(10, (100, 200), input_size)

# Training loop
epoch = 0
# for epoch in range(num_epochs):
total_loss = 0

# Shuffle the episodes at the beginning of each epoch
random.shuffle(episodes)

# Iterate over batches
# for batch_idx in range(0, len(episodes), batch_size):
batch_idx = 0
optimizer.zero_grad()

# Create the batch
batch_episodes = episodes[batch_idx:batch_idx + batch_size] # Select a maximum of batch_size episodes to be in the current batch

# Pad the batch if it's smaller than batch_size
while len(batch_episodes) < batch_size:
    batch_episodes += episodes[:batch_size - len(batch_episodes)]

batch_data = []
for episode in batch_episodes:
    # Choose a random start position
    start_position = random.randint(0, len(episode) - sequence_length - 1)

    # Extract the segment
    segment = episode[start_position:start_position + sequence_length + 1]
    batch_data.append(segment)
# Convert the batch data to a tensor
batch_data = torch.stack(batch_data).to(device) # (batch_size, sequence_length, input_size)
batch_data.shape
# Encode the warmup sequence
encoder_output = encoder(batch_data)  # (batch_size, sequence_length, embedding_size)
encoder_output.shape

# Discard the warmup period's output
predictor_input = encoder_output[:, warmup_length:-1,]  # (batch_size, prediction_length, embedding_size)
predictor_input.shape

# Prepare the actions for the predictor
actions = batch_data[:, warmup_length:-1, 6:]  # (batch_size, prediction_length, action_size)
actions.shape
# Predict the embeddings for the prediction length
predicted_embeddings = predictor(predictor_input.reshape(-1, embedding_size), actions.reshape(-1, action_size))
predicted_embeddings.shape

predicted_embeddings_per_batch = predicted_embeddings.reshape(batch_size, num_predictions, embedding_size)
predicted_embeddings_per_batch.shape

# Calculate the target embeddings
target_embeddings = encoder_output[:, warmup_length+1:]

# Calculate the loss
loss = loss_fn(predicted_embeddings, target_embeddings)

# Backpropagate and update the parameters
loss.backward()
optimizer.step()

total_loss += loss.item()

# Print the average loss for this epoch
print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(episodes):.4f}")
