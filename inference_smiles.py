import os
import PCL
import torch
import encoders
import decoders
import warnings
import base_model
import numpy as np
import transformers
import pandas as pd
from torch import nn
from constants import *
from pytorch_lamb import Lamb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
warnings.filterwarnings('ignore')


class SMILESGenerator(base_model.Model):
    def __init__(self, vocab_size, max_seq_length, embed_dim):
        super(SMILESGenerator, self).__init__()
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.embed_dim = embed_dim
        self.smiles_decoder = decoders.SMILESDecoder(self.vocab_size, self.max_seq_length, self.embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.vocab_size),
            nn.Softmax(dim=-1),
        )
        
    def forward(self, input_, memory):
        # Get the decoded vectors from smiles decoder
        output = self.smiles_decoder(input_, memory)
        # Add fully connected layer along with softmax
        output = self.classifier(output)
        return output


if __name__ == "__main__":
    
    # Loading the dataset
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


    # Create a general dataset to be used during training
    smiles_protein_dataset = SMILESProteinDataset(smiles_train, smiles_attention_mask_train, protein_train, protein_attention_mask_train, affinity_train)
    dataloader = DataLoader(smiles_protein_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    # Convert the validation sets to tensors
    smiles_validation, smiles_attention_mask_validation, protein_validation, protein_attention_mask_validation, affinity_validation = torch.tensor(smiles_validation).to(device), torch.tensor(smiles_attention_mask_validation).to(device), torch.tensor(protein_validation).to(device), torch.tensor(protein_attention_mask_validation).to(device), torch.tensor(affinity_validation).to(device)
    
    # Instantiate from Protein Encoder class
    protein_encoder = encoders.ProteinEncoder(protein_vocab_size, protein_max_length, embed_dim, latent_dim).to(device)

    # Load the pre-trained Protein Encoder
    if os.path.exists(protein_encoder_path):
        print("Loading the protein encoder model", flush=True)
        protein_encoder.load_state_dict(torch.load(protein_encoder_path))
    
    # # Freeze the layer of Protein Encoder
    for parameter in protein_encoder.parameters():
        parameter.requires_grad = False
    # list(protein_encoder.parameters())[-1].requires_grad = True
    
    
    # #Define SMILES and Protein encoder
    # smiles_encoder = encoders.SMILESEncoder(smiles_vocab_size, smiles_max_length, embed_dim, latent_dim).to(device)
    # protein_encoder = encoders.ProteinEncoder(protein_vocab_size, protein_max_length, embed_dim, latent_dim).to(device)
    
    # #Define Contrastive Learning model
    # model_cl = PCL.ContrastiveLearning(smiles_encoder, protein_encoder).to(device)
    
    # if torch.cuda.device_count() > 1:
    #     model_cl = torch.nn.DataParallel(model_cl)

    # """When there is a pre-trained model, uncomment the following code """
    # model_cl_path = './model.pth'
    # if os.path.exists(model_cl_path):
    #     print("Loading the model", flush=True)
    #     model_cl.load_state_dict(torch.load(model_cl_path))
    # for parameter in protein_encoder.parameters():
    #     parameter.requires_grad = False
        
    # Define smiles start toekn
    smiles_start_token = torch.tensor(smiles_tokenizer.cls_token_id).reshape(-1,1)
    
    # Instantiate from smiles generator
    smiles_generator = SMILESGenerator(smiles_vocab_size, smiles_max_length, latent_dim).to(device)
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        smiles_generator = torch.nn.DataParallel(smiles_generator)
    
    #Define optimizer
    optimizer = torch.optim.AdamW(smiles_generator.parameters(), lr=1e-3)
    # optimizer = Lamb(smiles_generator.parameters(), lr=1e-3)
    
    """When you want to use learning rate scheduler, incomment the following code"""
    # Define learning rate scheduler
    num_training_steps = len(dataloader) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    # Initialize the scheduler
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=5  # Number of cycles (restarts)
    )
    
    # Enable cudnn benchmark
    torch.backends.cudnn.benchmark = True
    
    # assert False, "Everything works well"
    
    print("Training Started...", flush=True)
    loss_list = []
    for epoch in range(epochs):
        smiles_generator.train()
        total_loss = 0
        for input_ in dataloader:
            smiles, _, protein, protein_attention_mask, _  = input_
            # smiles, smiles_attention_mask, protein, protein_attention_mask, affinity =  smiles.to(device), smiles_attention_mask.to(device), protein.to(device), protein_attention_mask.to(device), affinity.to(device)
            smiles, protein, protein_attention_mask = smiles.to(device), protein.to(device), protein_attention_mask.to(device)
            
            optimizer.zero_grad()
            
            generated_smiles, generated_smiles_vectors = initialize_prediction(len(protein), smiles_max_length, smiles_vocab_size, smiles_start_token)
            generated_smiles, generated_smiles_vectors = generated_smiles.to(device), generated_smiles_vectors.to(device)
            # Calculate the encoded protein
            with torch.no_grad():
                memory  = protein_encoder(protein, protein_attention_mask)
            
            for i in range(smiles_max_length - 1):
                prediction = smiles_generator(generated_smiles, memory)
                # prediction = torch.mul(prediction[:, -1, :], 1e6)
                prediction = prediction[:, -1, :]
                generated_smiles[:, i+1] = prediction.argmax(-1).detach()  # Avoid in-place operation
                generated_smiles_vectors[:, i+1, :] = prediction.detach()  # Avoid in-place operation
                # print(i, flush=True)
            
            # assert False, generated_smiles_vectors.size()
            # generated_smiles_vectors = torch.divide(generated_smiles_vectors.float(), 1e6).to(device).requires_grad_()
            # loss = torch.nn.functional.cross_entropy(generated_smiles_vectors, smiles)
            generated_smiles_vectors = generated_smiles_vectors.requires_grad_()
            loss = torch.nn.functional.cross_entropy(generated_smiles_vectors.view(-1, smiles_vocab_size), smiles.view(-1))
            loss.backward()
            optimizer.step()
            
            """When you want to use learning rate scheduler, incomment the following code"""
            scheduler.step()
            
            total_loss += loss.item()
            torch.cuda.empty_cache()

        loss_list.append(total_loss / len(dataloader))
        print(f"Epoch {epoch+1}'s training Loss: {total_loss / len(dataloader)}", flush=True)

        if ((epoch+1)%5==0):
            print(f"The last 5 epochs' Loss: {sum(loss_list[-5:]) /5}", flush=True)
            torch.save(smiles_generator.state_dict(), './decoder.pth')
            torch.save(protein_encoder.state_dict(), './findal_decoder.pth')
            torch.save(loss_list, 'decoder_loss')


    
                
            
            
            
            
            
            
            

    