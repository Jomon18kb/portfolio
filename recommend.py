import torch
import pandas as pd
import joblib
import os

from model import NCFModel

def recommend_movies(user_id, top_n=10):
    print(f"Generating recommendations for User {user_id}...")
    
    # Load encoders
    try:
        user_encoder = joblib.load('models/user_encoder.pkl')
        item_encoder = joblib.load('models/item_encoder.pkl')
    except FileNotFoundError:
        print("Model or encoders not found. Please run train.py first.")
        return
        
    # Check if user exists in the dataset
    if user_id not in user_encoder.classes_:
        print(f"User {user_id} not found in the training data.")
        return
        
    encoded_user = user_encoder.transform([user_id])[0]
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NCFModel(num_users=num_users, num_items=num_items, embedding_dim=32, hidden_layers=[64, 32, 16])
    model.load_state_dict(torch.load('models/ncf_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # Load dataset to see what the user has already rated
    ratings_file = './data/ml-latest-small/ratings.csv'
    movies_file = './data/ml-latest-small/movies.csv'
    
    ratings_df = pd.read_csv(ratings_file)
    movies_df = pd.read_csv(movies_file)
    
    user_seen_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
    all_movies = ratings_df['movieId'].unique()
    
    # Filter out unseen movies
    unseen_movies = [m for m in all_movies if m not in user_seen_movies]
    
    # Convert unseen movies to their encoded indices
    valid_unseen_movies = [m for m in unseen_movies if m in item_encoder.classes_]
    encoded_items = item_encoder.transform(valid_unseen_movies)
    
    # Predict
    user_tensor = torch.tensor([encoded_user] * len(encoded_items), dtype=torch.long).to(device)
    item_tensor = torch.tensor(encoded_items, dtype=torch.long).to(device)
    
    with torch.no_grad():
        predictions = model(user_tensor, item_tensor).cpu().numpy()
        
    # Get top N recommendations
    recommendation_df = pd.DataFrame({
        'movieId': valid_unseen_movies,
        'predicted_rating': predictions
    })
    
    recommendation_df = recommendation_df.sort_values(by='predicted_rating', ascending=False).head(top_n)
    
    # Merge with movie titles
    recommendation_df = recommendation_df.merge(movies_df, on='movieId', how='left')
    
    print("\n-------------------------------------------")
    print(f"Top {top_n} Movie Recommendations for User {user_id}:")
    print("-------------------------------------------")
    for i, row in recommendation_df.iterrows():
        print(f"{i+1}. {row['title']} (Genres: {row['genres']}) - Predicted Rating: {row['predicted_rating']:.2f}")
    print("-------------------------------------------\n")

if __name__ == '__main__':
    # Try recommending for user 1
    # Note: user 1 is present in the ml-latest-small dataset.
    recommend_movies(user_id=2, top_n=10)
