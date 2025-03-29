# this method is intractable for large n

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import boto3

from sklearn.metrics.pairwise import cosine_similarity
from numpy.random import choice, seed
from sklearn.manifold import TSNE

user_df = pd.read_csv('/Users/hsp/Downloads/synthetic_user_data.csv')
user_df = user_df.drop(columns=['Unnamed: 0'])

def fetch_dynamoDB_data():
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('users-table')
    response = table.scan()
    items = response['Items']
    dynamo_df = pd.DataFrame(items)
    
    return dynamo_df

dynamo_df = fetch_dynamoDB_data()
def transform_to_one_hot(dynamo_df):
    
    dynamo_df = dynamo_df.drop(columns=['biography', 'condition', 'lat', 'long'])
    one_hot_dynamo = pd.DataFrame()
    one_hot_dynamo['Name'] = dynamo_df['name']
    
    for question in range(1, 21):
        for option in range(1, 4):
            column_name = f'Q{question}_Option{option}'
            
            answer_col = f'answer{question:02d}'
            
            if answer_col in dynamo_df.columns:
                one_hot_dynamo[column_name] = (dynamo_df[answer_col] == option).astype(int)
            else:
                one_hot_dynamo[column_name] = 0
                print(f"Column '{answer_col}' not found in DataFrame")
    return one_hot_dynamo

one_hot_dynamo = transform_to_one_hot(dynamo_df)
# print(one_hot_dynamo.head())

combined_df = pd.concat([user_df, one_hot_dynamo], axis=0,
ignore_index=True)

user_names = combined_df['Name'].values
features = combined_df.drop('Name', axis=1).values

# dimensionality reduction
tsne = TSNE(n_components=3,
            perplexity=49,
            random_state=42,
            learning_rate='auto',
            max_iter=10000
)

tsne_results = tsne.fit_transform(features)
tsne_df = pd.DataFrame(tsne_results)
tsne_df.insert(0, 'Name', user_names)

#print(tsne_df.head())
#print(f"t-SNE results shape: {tsne_df.shape}")

def calculate_similarities(tsne_df, username):
    if username not in tsne_df['Name'].values:
        raise ValueError(f"'{username}' not found in database.")
    
    target_user_index = tsne_df[tsne_df['Name'] == username].index[0]
    target_user_data = tsne_df.iloc[target_user_index, 1:].values
    
    names = []
    # jaccard_scores = []
    cosine_scores = []
    
    for index, row in tsne_df.iterrows():
        current_name = row['Name']
        if current_name == username:
            continue
        
        current_user_data = row.iloc[1:].values
        
        # cosine similarity
        cos_sim = cosine_similarity(
            target_user_data.reshape(1, -1), 
            current_user_data.reshape(1, -1)
        )[0][0]
        
        names.append(current_name)
        # jaccard_scores.append(jaccard)
        cosine_scores.append(cos_sim)
        
        # avg_similarity = (jaccard + cos_sim) / 2 * 100
        
        results = pd.DataFrame({
        'Name': names,
        #'Jaccard_Similarity': jaccard_scores,
        'Cosine_Similarity': cosine_scores,
        #'Average_Similarity': avg_similarity,
    })
        
    results = results.sort_values('Cosine_Similarity', ascending=False)
    return results

def find_most_similar_users(tsne_df, username, top_n=5):
        similarities = calculate_similarities(tsne_df, username)
        return similarities.head(top_n)
    
# example 
top_similar = find_most_similar_users(tsne_df, "Parker", top_n=3)
print(top_similar)

