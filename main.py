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
        self.ratings_df = pd.read_csv('ratings.csv')

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

    def __getitem__(self, item):
        user_idx = self.ratings_df.iloc[idx]['user_idx']
        movie_idx = self.ratings_df.iloc[idx]['movie_idx']
        rating = self.ratings_df.iloc[idx]['rating']

        sample = {
            'user_idx': user_idx
            'movie_idx': movie_idx
            'rating': rating
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

class MatrixFactorisation(nn.Module):
    def __init__(self, n_users, n_items, n_factors=50):
        super().__init()

        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)

        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)

    def forward(self, user, item):
        u = self.user_factors(user)
        v = self.item_factors(item)

        rating = torch.sum(u * v, dim=1)
        return rating

