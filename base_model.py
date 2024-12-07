import os
import torch
from torch import nn

#Define a base model including the common methods among the future models
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def load(self, path):
        if os.path.exists(path):
            print('The model is loading ...', flush=True)
            self.load_state_dict(torch.load(path))
        else:
            assert False, "The path doesn't exist"
    
    def save(self, path):
        torch.save(self.state_dict(), path)


if __name__ == "__main__":
    # Test the base model
    model = Model()
    model.save('./test_model.pth')
    model.load('./test_model.pth')
    print(model, flush=True)
    
    # Test the postional encoder
    # Parameters
    batch_size = 2
    seq_length = 5
    embed_dim = 8
    max_seq_length = 10

    # Create a dummy input tensor (batch_size, seq_length, embed_dim)
    x = torch.rand(batch_size, seq_length, embed_dim)
    pos_encoder = PositionalEncoder(max_seq_length, embed_dim)
    encoded_x = pos_encoder(x)
    print(encoded_x)
    

