"""Probabilistic Contrastive Learning"""
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

def contrastive_loss(xs, xp, affinity, affinity_alpha=0.05):
    
    if len(xs.size()) == 3:
        xs, _ = torch.max(xs, dim=1)
        xp, _ = torch.max(xp, dim=1)
        # xs= torch.mean(xs, dim=1)
        # xp= torch.mean(xp, dim=1)
    
    logits = torch.matmul(xs, xp.T).to(xs.device)
    
    # Create labels for cross-entropy
    labels = torch.arange(xs.size(0)).to(xs.device)

    # Compute affinity loss using sigmoid and log (with clamping to avoid numerical issues)
    affinity_loss = torch.clamp(torch.sigmoid(affinity), min=1e-7, max=1.0).to(xs.device)
    
    # Compute cross-entropy loss
    cross_entropy_loss = torch.nn.functional.cross_entropy(logits, labels,  reduction='none')
    
    #Regularization section
    reg_section = torch.mul(cross_entropy_loss, affinity_loss)
    
    # Combine both losses
    total_loss = torch.add(cross_entropy_loss, torch.mul(affinity_alpha, reg_section))
    
    return torch.mean(total_loss)



class ContrastiveLearning(base_model.Model):
    def __init__(self, smiles_encoder, protein_encoder):
        super(ContrastiveLearning, self).__init__()
        self.smiles_encoder = smiles_encoder
        self.protein_encoder = protein_encoder
        self.smiles_fc = torch.nn.Sequential(
            torch.nn.Linear(self.smiles_encoder.latent_dim, self.smiles_encoder.latent_dim),
            torch.nn.Softmax(dim=-1)
        )
        self.protein_fc = torch.nn.Sequential(
            torch.nn.Linear(self.protein_encoder.latent_dim, self.protein_encoder.latent_dim),
            torch.nn.Softmax(dim=-1)
        )
    
    def forward(self, xs, xs_attention_mask, xp, xp_attention_mask):
        xs = self.smiles_encoder(xs, xs_attention_mask)
        xp = self.protein_encoder(xp, xp_attention_mask)
        
        xs = self.smiles_fc(xs)
        xp = self.protein_fc(xp)
        
        return xs, xp
                
if __name__ == "__main__": 
    np.random.seed(2024)
    
    # Binding Database
    print('Loading the dataset', flush=True)
    bindingDB_data = BindingDB()
    
    # Split dataset into train, validation, and test sets
    dataset_length = len(bindingDB_data.smiles_list)
    validation_test_indices = torch.randint(0, dataset_length, (int(dataset_length * 5e-2),))
    validation, test = validation_test_indices[:len(validation_test_indices)//2], validation_test_indices[len(validation_test_indices)//2:]
    
    # Create boolean mask for training indices
    mask = torch.ones(dataset_length, dtype=torch.bool)
    mask[validation_test_indices] = False
    
    # Tokenize the smiles and protein sequences
    if tokenized_data:
        print('Loading the tokenized data', flush=True)
        smiles_train, smiles_attention_mask_train = torch.load(os.path.join(tokenized_data_path, 'smiles_train'))
        smiles_validation, smiles_attention_mask_validation = torch.load(os.path.join(tokenized_data_path, 'smiles_validation'))
        smiles_test, smiles_attention_mask_test = torch.load(os.path.join(tokenized_data_path, 'smiles_test'))
        
        protein_train, protein_attention_mask_train = torch.load(os.path.join(tokenized_data_path, 'protein_train'))
        protein_validation, protein_attention_mask_validation = torch.load(os.path.join(tokenized_data_path, 'protein_validation'))
        protein_test, protein_attention_mask_test = torch.load(os.path.join(tokenized_data_path, 'protein_test'))

        affinity_train = torch.load(os.path.join(tokenized_data_path, 'affinity_train'))
        affinity_validation = torch.load(os.path.join(tokenized_data_path, 'affinity_validation'))
        affinity_test = torch.load(os.path.join(tokenized_data_path, 'affinity_test'))
        
    else:
        affinity_train = bindingDB_data.affinity_list[mask]
        affinity_validation = bindingDB_data.affinity_list[validation]
        affinity_test = bindingDB_data.affinity_list[test]
        
        print('Tokenizing the dataset', flush=True)
        smiles, smiles_attention_mask = tokenizer(bindingDB_data.smiles_list, smiles_tokenizer, smiles_max_length)
        protein, protein_attention_mask = tokenizer(bindingDB_data.protein_list, protein_tokenizer, protein_max_length,)
        
        smiles_train = smiles[mask]
        smiles_validation = smiles[validation]
        smiles_test = smiles[test]
        
        smiles_attention_mask_train = smiles_attention_mask[mask]
        smiles_attention_mask_validation = smiles_attention_mask[validation]
        smiles_attention_mask_test = smiles_attention_mask[test]
        
        protein_train = protein[mask]
        protein_validation = protein[validation]
        protein_test = protein[test]

        protein_attention_mask_train = protein_attention_mask[mask]
        protein_attention_mask_validation = protein_attention_mask[validation]
        protein_attention_mask_test = protein_attention_mask[test]
        
        torch.save((smiles_train, smiles_attention_mask_train), os.path.join(tokenized_data_path, 'smiles_train'))
        torch.save((smiles_validation, smiles_attention_mask_validation), os.path.join(tokenized_data_path, 'smiles_validation'))
        torch.save((smiles_test, smiles_attention_mask_test), os.path.join(tokenized_data_path, 'smiles_test'))
        
        torch.save((protein_train, protein_attention_mask_train), os.path.join(tokenized_data_path, 'protein_train'))
        torch.save((protein_validation, protein_attention_mask_validation), os.path.join(tokenized_data_path, 'protein_validation'))
        torch.save((protein_test, protein_attention_mask_test), os.path.join(tokenized_data_path, 'protein_test'))
        
        torch.save(affinity_train, os.path.join(tokenized_data_path, 'affinity_train'))
        torch.save(affinity_validation, os.path.join(tokenized_data_path, 'affinity_validation'))
        torch.save(affinity_test, os.path.join(tokenized_data_path, 'affinity_test'))

    # Create a general dataset to be used during training
    smiles_protein_dataset = SMILESProteinDataset(smiles_train, smiles_attention_mask_train, protein_train, protein_attention_mask_train, affinity_train)
    dataloader = DataLoader(smiles_protein_dataset, batch_size=batch_size, shuffle=True)
    
    smiles_validation, smiles_attention_mask_validation, protein_validation, protein_attention_mask_validation, affinity_validation = torch.tensor(smiles_validation).to(device), torch.tensor(smiles_attention_mask_validation).to(device), torch.tensor(protein_validation).to(device), torch.tensor(protein_attention_mask_validation).to(device), torch.tensor(affinity_validation).to(device)
    
    #Define SMILES and Protein encoder
    smiles_encoder = encoders.SMILESEncoder(smiles_vocab_size, smiles_max_length, embed_dim, latent_dim).to(device)
    protein_encoder = encoders.ProteinEncoder(protein_vocab_size, protein_max_length, embed_dim, latent_dim).to(device)

    #Define Contrastive Learning model
    model = ContrastiveLearning(smiles_encoder, protein_encoder).to(device)
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = torch.nn.DataParallel(model)

    """When there is a pre-trained model, uncomment the following code """
    model_path = './model.pth'
    if os.path.exists(model_path):
        print("Loading the model", flush=True)
        model.load_state_dict(torch.load(model_path))
    
    #Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # optimizer = Lamb(model.parameters(), lr=1e-3)
    
    """When you want to use learning rate scheduler, incomment the following code"""
    # # Define learning rate scheduler
    num_training_steps = len(dataloader) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    # Initialize the scheduler
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=5  # Number of cycles (restarts)
    )
    
    print("Training Started...", flush=True)
    loss_list = []
    loss_list_validation = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for input_ in dataloader:
            smiles, smiles_attention_mask, protein, protein_attention_mask, affinity = input_
            smiles, smiles_attention_mask, protein, protein_attention_mask, affinity =  smiles.to(device), smiles_attention_mask.to(device), protein.to(device), protein_attention_mask.to(device), affinity.to(device)
            optimizer.zero_grad()
            xs, xp = model(smiles, smiles_attention_mask, protein, protein_attention_mask)
            loss = contrastive_loss(xs, xp, affinity)
            loss.backward()
            optimizer.step()
            
            """When you want to use learning rate scheduler, incomment the following code"""
            scheduler.step()
            
            total_loss += loss.item()
            torch.cuda.empty_cache()
        
        loss_list.append(total_loss / len(dataloader))
        with torch.no_grad():
            model.eval()
            xs, xp = model(smiles_validation, smiles_attention_mask_validation, protein_validation, protein_attention_mask_validation)
            loss_validation = contrastive_loss(xs, xp, affinity_validation)
            loss_list_validation.append(loss_validation.item())
            print(f"Epoch {epoch+1}'s training Loss: {total_loss / len(dataloader)}, validation loss {loss_validation.item()}", flush=True)
            
        if ((epoch+1)%5==0):
            model.train()
            print(f"The last 5 epochs' Loss: {sum(loss_list[-5:]) /5}", flush=True)
            torch.save(model.state_dict(), model_path)
            torch.save(model.module.smiles_encoder.state_dict(), smiles_encoder_path)
            torch.save(model.module.protein_encoder.state_dict(), protein_encoder_path)
            torch.save(loss_list, './loss')
            torch.save(loss_list_validation, './loss_validation')
            
            
    
    
    