import troch
import torch.nn as nn
import torch.optim as optim
from neo import ModelConfig, Neo


config = ModelConfig()
model = Neo(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloader, epochs, learning_rate, device, compile_model=False):
  model.to(device)
  if compile_model:
    model = torch.compile(model)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(epochs):
      model.train()  
      total_loss = 0
      for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch, start_pos = 0)
        loss = criterion(output[:, :-1].reshape(-1, config.vocab_size), batch[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
      avg_loss = total_loss/ len(dataloader)
      print(f"Epoch: {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")
