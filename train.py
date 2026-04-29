import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import joblib

from data_loader import load_and_preprocess_data
from model import NCFModel

class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.users = torch.tensor(data['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(data['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(data['rating'].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

def train_model():
    print("Loading and preparing data...")
    train_df, val_df, num_users, num_items, user_encoder, item_encoder = load_and_preprocess_data()
    
    # Save encoders for later use during recommendation
    os.makedirs('models', exist_ok=True)
    joblib.dump(user_encoder, 'models/user_encoder.pkl')
    joblib.dump(item_encoder, 'models/item_encoder.pkl')
    
    batch_size = 512
    
    train_dataset = MovieLensDataset(train_df)
    val_dataset = MovieLensDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = NCFModel(num_users=num_users, num_items=num_items, embedding_dim=32, hidden_layers=[64, 32, 16])
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        for users, items, ratings in train_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(users, items)
            loss = criterion(predictions, ratings)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * len(users)
            
        avg_train_loss = total_train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for users, items, ratings in val_loader:
                users, items, ratings = users.to(device), items.to(device), ratings.to(device)
                predictions = model(users, items)
                loss = criterion(predictions, ratings)
                total_val_loss += loss.item() * len(users)
                
        avg_val_loss = total_val_loss / len(val_dataset)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")
        
    print("Training complete. Saving model...")
    torch.save(model.state_dict(), 'models/ncf_model.pth')
    
    # Plotting training curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/loss_curve.png')
    print("Model and loss curve saved in 'models/' directory.")

if __name__ == '__main__':
    train_model()
