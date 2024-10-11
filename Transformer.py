import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import pad, one_hot
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return both the feature and the corresponding label
        return self.features[idx], self.labels[idx]
    
def create_padding_mask(tensor_input):
    mask = (tensor_input.sum(dim=2) == 0).float()
    return mask
    

# Define the Transformer model for classification
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dim_feedforward, dropout=0.15):

        super(TransformerClassifier, self).__init__()
        self.model_dim = d_model
        self.projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead = num_heads, dropout = dropout, dim_feedforward=dim_feedforward,batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers,enable_nested_tensor=False)
        #self.layer_norm = nn.LayerNorm(d_model)
        self.classification_head = nn.Linear(d_model, 1)


    def forward(self, x):

        x = self.projection(x) * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x,is_causal = False, src_key_padding_mask = create_padding_mask(x) ) 
        #x = self.layer_norm(x)
        x = x.mean(dim=1)
        
        x = self.classification_head(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=650):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1)].requires_grad_(False)
        
        return self.dropout(x)

