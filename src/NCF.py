import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from datetime import datetime


def NCF_calculation(df):
    np.random.seed(123)

    df['rank_latest'] = df.groupby(['userId'])['timestamp'] \
                                .rank(method='first', ascending=False)

    print(df)

    #Using earlier review for trainning, and using latest review for testing
    train_ratings = df[df['rank_latest'] != 1]
    test_ratings = df[df['rank_latest'] == 1]
    
    #Converting the dataset into an implicit feedback dataset
    #Binarize the ratings to 1 means the user has interacted with the movie
    train_ratings.loc[:, 'rating'] = 1
    # Generate negative samples to train our models
    # Get a list of all movie IDs
    all_movieIds = df['movieId'].unique()

    # Placeholders that will hold the training data
    users, items, labels = [], [], []

    # This is the set of items that each user has interaction with
    user_item_set = set(zip(train_ratings['userId'], train_ratings['movieId']))

    # 4:1 ratio of negative to positive samples
    num_negatives = 4

    for (u, i) in tqdm(user_item_set):
        users.append(u)
        items.append(i)
        labels.append(1) # items that the user has interacted with are positive
        for _ in range(num_negatives):
            # randomly select an item
            negative_item = np.random.choice(all_movieIds) 
            # check that the user has not interacted with this item
            while (u, negative_item) in user_item_set:
                negative_item = np.random.choice(all_movieIds)
            users.append(u)
            items.append(negative_item)
            labels.append(0) # items not interacted with are negative
            
    class MovieLensTrainDataset(Dataset):
        """MovieLens PyTorch Dataset for Training
        
        Args:
            ratings (pd.DataFrame): Dataframe containing the movie ratings
            all_movieIds (list): List containing all movieIds
        
        """

        def __init__(self, ratings, all_movieIds):
            self.users, self.items, self.labels = self.get_dataset(ratings, all_movieIds)

        def __len__(self):
            return len(self.users)
    
        def __getitem__(self, idx):
            return self.users[idx], self.items[idx], self.labels[idx]

        def get_dataset(self, ratings, all_movieIds):
            users, items, labels = [], [], []
            user_item_set = set(zip(ratings['userId'], ratings['movieId']))

            num_negatives = 4
            for u, i in user_item_set:
                users.append(u)
                items.append(i)
                labels.append(1)
                for _ in range(num_negatives):
                    negative_item = np.random.choice(all_movieIds)
                    while (u, negative_item) in user_item_set:
                        negative_item = np.random.choice(all_movieIds)
                    users.append(u)
                    items.append(negative_item)
                    labels.append(0)

            return torch.tensor(users), torch.tensor(items), torch.tensor(labels)


    class NCF(pl.LightningModule):
        """ Neural Collaborative Filtering (NCF)
        
            Args:
                num_users (int): Number of unique users
                num_items (int): Number of unique items
                ratings (pd.DataFrame): Dataframe containing the movie ratings for training
                all_movieIds (list): List containing all movieIds (train + test)
        """
        
        def __init__(self, num_users, num_items, ratings, all_movieIds):
            super().__init__()
            self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
            self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
            self.fc1 = nn.Linear(in_features=16, out_features=64)
            self.fc2 = nn.Linear(in_features=64, out_features=32)
            self.output = nn.Linear(in_features=32, out_features=1)
            self.ratings = ratings
            self.all_movieIds = all_movieIds
            
        def forward(self, user_input, item_input):
            
            # Pass through embedding layers
            user_embedded = self.user_embedding(user_input)
            item_embedded = self.item_embedding(item_input)

            # Concat the two embedding layers
            vector = torch.cat([user_embedded, item_embedded], dim=-1)

            # Pass through dense layer
            vector = nn.ReLU()(self.fc1(vector))
            vector = nn.ReLU()(self.fc2(vector))

            # Output layer
            pred = nn.Sigmoid()(self.output(vector))

            return pred
        
        def training_step(self, batch, batch_idx):
            user_input, item_input, labels = batch
            predicted_labels = self(user_input, item_input)
            loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters())

        def train_dataloader(self):
            return DataLoader(MovieLensTrainDataset(self.ratings, self.all_movieIds),
                            batch_size=512, num_workers=1)

        
    num_users = df['userId'].max()+1
    num_items = df['movieId'].max()+1

    all_movieIds = df['movieId'].unique()

    model = NCF(num_users, num_items, train_ratings, all_movieIds)
    trainer = pl.Trainer(max_epochs=3, gpus=1, reload_dataloaders_every_epoch=True,
                     progress_bar_refresh_rate=50, logger=False, checkpoint_callback=False)

    trainer.fit(model)

    # User-item pairs for testing
    test_user_item_set = set(zip(test_ratings['userId'], test_ratings['movieId']))

    # Dict of all items that are interacted with by each user
    user_interacted_items = df.groupby('userId')['movieId'].apply(list).to_dict()

    hits = []
    for (u,i) in tqdm(test_user_item_set):
        interacted_items = user_interacted_items[u]
        not_interacted_items = set(all_movieIds) - set(interacted_items)
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
        test_items = selected_not_interacted + [i]
        
        predicted_labels = np.squeeze(model(torch.tensor([u]*100), 
                                            torch.tensor(test_items)).detach().numpy())
        
        top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]
        
        if i in top10_items:
            hits.append(1)
        else:
            hits.append(0)
            
    print("The Hit Ratio @ 10 is {:.2f}".format(np.average(hits)))