import os
import zipfile
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
data_dir = './data'
zip_path = os.path.join(data_dir, 'ml-latest-small.zip')
extracted_dir = os.path.join(data_dir, 'ml-latest-small')
ratings_file = os.path.join(extracted_dir, 'ratings.csv')

def download_and_extract_data():
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    if not os.path.exists(zip_path):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
        
    if not os.path.exists(extracted_dir):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")

def load_and_preprocess_data():
    download_and_extract_data()
    
    print(f"Loading data from {ratings_file}...")
    df = pd.read_csv(ratings_file)
    
    # We only need user, item, and rating for collaborative filtering
    # df = df[['userId', 'movieId', 'rating']]
    
    # Map users and items to contiguous indices
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    df['user_id'] = user_encoder.fit_transform(df['userId'])
    df['item_id'] = item_encoder.fit_transform(df['movieId'])
    
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    
    print(f"Number of distinct users: {num_users}")
    print(f"Number of distinct items: {num_items}")
    
    # Split into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    return train_df, val_df, num_users, num_items, user_encoder, item_encoder

if __name__ == '__main__':
    load_and_preprocess_data()
