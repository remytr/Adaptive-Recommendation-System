### Recommendation System with Reinforcement Learning ###

# Import libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests
import zipfile
import io
import random
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class movieLensDataset(Dataset):
    def __init__(self, ratings_file, transform=None):
        self.ratings_df = pd.read_csv("ml-latest-small/ratings.csv")

        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()

        self.user_encoder.fit(self.ratings_df['userId'])
        self.movie_encoder.fit(self.ratings_df['movieId'])

        self.ratings_df['user_idx'] = self.user_encoder.transform(self.ratings_df['userId'])
        self.ratings_df['movie_idx'] = self.movie_encoder.transform(self.ratings_df['movieId'])

        self.n_users = len(self.user_encoder.classes_)
        self.n_movies = len(self.movie_encoder.classes_)

        self.transform = transform

    def __len__(self):
        return len(self.ratings_df)

    def __getitem__(self, idx):
        user_idx = self.ratings_df.iloc[idx]['user_idx']
        movie_idx = self.ratings_df.iloc[idx]['movie_idx']
        rating = self.ratings_df.iloc[idx]['rating']

        sample = {
            'user_idx': int(user_idx), # Change to int as was coming in as DoubleTensor (Floats).
            'movie_idx': int(movie_idx),
            'rating': float(rating)
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

def prepare_data():
    dataset = movieLensDataset("ml-latest-small/ratings.csv")

    train_size = 0.7
    val_size = 0.15
    test_size = 0.15

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    train_split = int(np.floor(train_size * len(dataset)))
    val_split = int(np.floor((train_size + val_size)* len(dataset)))

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    train_dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers = 4,
        sampler=train_indices
    )

    val_dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        sampler=val_indices
    )

    test_dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        # num_workers - watch RAM
        sampler=test_indices
    )

    return dataset, train_dataloader, val_dataloader, test_dataloader

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=50):
        super().__init__()

        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)

        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)

    def forward(self, user, item):
        u = self.user_factors(user)
        v = self.item_factors(item)

        rating = torch.sum(u * v, dim=1)
        return rating

def train_matrix_factorization(dataset, train_dataloader, val_dataloader, epochs =10):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    model = MatrixFactorization(dataset.n_users, dataset.n_movies).to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            user_idx = batch['user_idx'].to(device)
            movie_idx = batch['movie_idx'].to(device)
            # moves from CPU or GPU. Will be CPU since no GPU
            rating = batch['rating'].float().to(device)

            optimizer.zero_grad()

            pred = model(user_idx, movie_idx)

            loss = criterion(pred, rating)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                user_idx = batch['user_idx'].to(device)
                movie_idx = batch['movie_idx'].to(device)
                rating = batch['rating'].float().to(device)

                # Forward pass
                pred = model(user_idx, movie_idx)

                # Compute loss
                loss = criterion(pred, rating)
                running_val_loss += loss.item()

        val_loss = running_val_loss / len(val_dataloader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model, train_losses, val_losses


# Main function
def main():
    print("Preparing...")
    dataset, train_dataloader, val_dataloader, test_dataloader = prepare_data()

    # Train matrix factorization model
    print("\nTraining matrix factorization model...")
    mf_model, train_losses, val_losses = train_matrix_factorization(
        dataset, train_dataloader, val_dataloader, epochs=10
    )

    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Matrix Factorization Training Progress')
    plt.legend()
    plt.savefig('mf_training.png')
    plt.show()

if __name__ == "__main__":
    main()
