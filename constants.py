import os
import torch
import encoders
import warnings
import base_model
import numpy as np
import transformers
import pandas as pd
from constants import *
from pytorch_lamb import Lamb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
warnings.filterwarnings('ignore')

"""Parameters"""
epochs = 500 
batch_size = 256
smiles_max_length = 50
protein_max_length = 512
embed_dim = 256
latent_dim = 128
tokenized_data = True
# One the following items can be selected as affinity target: ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)', 'kon (M-1-s-1)', 'koff (s-1)']
affinity_target_type = 'Kd (nM)'
model_path = './model.pth'
smiles_encoder_path = './smiles_encoder.pth'
protein_encoder_path = './protein_encoder.pth'
tokenized_data_path = './tokenized_data'  

# Chose a gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, flush=True)

# Save CPU
cpu = torch.device('cpu')

# Initialize the necessary tokenizers
print('Loading the tokenizers', flush=True)
smiles_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", do_lower_case=False)
protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

smiles_vocab_size = smiles_tokenizer.vocab_size
protein_vocab_size = protein_tokenizer.vocab_size


class BindingDB():
    def __init__(self, path='./BindingDB_All_202405.tsv', affinity_target_type=affinity_target_type, nrows=1000000):
        # One the following items can be selected as affinity target: ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)', 'kon (M-1-s-1)', 'koff (s-1)']
        self.df_bd = pd.read_csv(path, sep='\t', nrows=nrows)

        # Select the necessary features
        self.features = ['Ligand SMILES', 'BindingDB Target Chain Sequence', affinity_target_type]
        self.df_bd = self.df_bd[self.features]
        self.df_bd = self.df_bd.dropna().reset_index(drop=True)
        
        # Clean the affinity_target_type column from the sign of '<' or '>' and change its type to float32
        self.df_bd.loc[:,affinity_target_type] = self.df_bd[affinity_target_type].astype(str)
        self.df_bd[affinity_target_type] = self.df_bd[affinity_target_type].apply(lambda x: x[1:] if '>' in x or '<' in x else x).astype('float32')
        
        # Filter the rows having lower 30 nm affinity target
        self.df_bd = self.df_bd[self.df_bd[affinity_target_type] < 30]
        
        # Restructure the data of BindingDB's Target Chain Sequence
        self.df_bd.loc[:, 'BindingDB Target Chain Sequence'] = self.df_bd.loc[:,'BindingDB Target Chain Sequence'].apply(lambda x: " ".join(x))
        
        # Take out the necessary lists of data
        self.smiles_list = self.df_bd['Ligand SMILES'].to_numpy()
        self.protein_list = self.df_bd['BindingDB Target Chain Sequence'].to_numpy()
        self.affinity_list = self.df_bd[affinity_target_type].to_numpy()
        # self.affinity_list = torch.tensor(self.affinity_list, dtype=torch.float32)


class SMILESProteinDataset(Dataset):
    def __init__(self, smiles, smiles_attention_mask, protein, protein_attention_mask, affinity):
        self.smiles = smiles
        self.smiles_attention_mask = smiles_attention_mask
        self.protein = protein
        self.protein_attention_mask = protein_attention_mask
        self.affinity = affinity

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return (
            self.smiles[idx],
            self.smiles_attention_mask[idx],
            self.protein[idx],
            self.protein_attention_mask[idx],
            self.affinity[idx],
        )

def tokenizer(list_, tokenizer, max_length, padding='max_length', truncation=True, return_tensors='pt'):
    tokenized_data = []
    attention_masks = []
    for item in list_:
        encoded = tokenizer(item, max_length=max_length, padding=padding, truncation=truncation, return_tensors=return_tensors)
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        tokenized_data.append(input_ids)
        attention_masks.append(attention_mask)

    return np.array(tokenized_data), np.array(attention_masks)


def initialize_prediction(size, sequence_length, vocab_size, start_token):
    generated_data = torch.zeros((size, sequence_length), dtype=torch.int32)
    generated_data_vectors = torch.zeros((size, sequence_length, vocab_size)).float()
    generated_data[:, 0] = start_token
    generated_data_vectors[:, 0, start_token] = 1.0
    return generated_data, generated_data_vectors