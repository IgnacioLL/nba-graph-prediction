import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GNNRegression(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNRegression, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # First GCN layer
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)

        # Second GCN layer
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        
        # Global pooling (mean of node features for each graph in the batch)
        x = global_mean_pool(x, batch)
        
        # Linear layer for regression
        x = self.lin(x)
        
        return x
    

def train_nn(model, criterion, optimizer, loader, patience_early_stopping=7, patience_plateau=3, min_lr=1e-5, save_path='best_model.pth'):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using :", device)
    
    # Move model to GPU if available
    model.to(device)
    
    # Initialize the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_plateau, min_lr=min_lr)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(100):
        total_loss = 0
        for batch in loader:
            # Move data to GPU if available
            batch = batch.to(device)
            
            optimizer.zero_grad()
            output = model(batch)
            
            # Reshape target tensor to match the shape of the output tensor
            target = batch.y.view(-1, 1)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        
        # Step the scheduler
        scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            
            # Save the model
            torch.save(model.state_dict(), save_path)
            print(f'Model saved at epoch {epoch} with loss {avg_loss}')
        else:
            epochs_no_improve += 1
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss}')
        
        if epochs_no_improve >= patience_early_stopping:
            print(f'Early stopping at epoch {epoch}')
            break