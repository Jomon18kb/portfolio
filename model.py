import torch
import torch.nn as nn

class NCFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[128, 64, 32]):
        super(NCFModel, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Build MLP layers
        mlp_layers = []
        input_dim = embedding_dim * 2  # Concatenated user and item embeddings
        
        for hidden_dim in hidden_layers:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
            
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final output predicts the rating (regression)
        self.output_layer = nn.Linear(input_dim, 1)
        
    def forward(self, user_indices, item_indices):
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)
        
        # Concatenate user and item embeddings
        x = torch.cat([user_vec, item_vec], dim=-1)
        
        # Pass through MLP
        x = self.mlp(x)
        
        # Final prediction
        prediction = self.output_layer(x)
        return prediction.squeeze()
