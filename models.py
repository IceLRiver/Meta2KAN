import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.init as init



class BatteryTransformer(nn.Module):
    def __init__(self, input_dim=4, output_dim=1, d_model=128, nhead=4, num_layers=4, dim_feedforward=128, dropout=0.1):
        super(BatteryTransformer, self).__init__()
        
        # Embedding layer: Project input to d_model dimensions
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Linear regression head
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = x.unsqueeze(1)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        
        # Transformer Encoder: (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)
        
        x = x.squeeze(0)
        out = self.fc(x)
        
        return out

class BatteryCHR(nn.Module):
    def __init__(self, num_feature=4):
        super(BatteryCHR, self).__init__()
        self.model_soc = BatteryTransformer(num_feature+1)
        self.model_soh = BatteryTransformer(num_feature+1)
        self.model_Qr = BatteryTransformer(num_feature+1)

    def forward(self, x, soh_t0):
        soc = self.model_soc(torch.cat([x, soh_t0], dim=1))
        soh = self.model_soh(torch.cat([x, soc], dim=1))
        Qr = self.model_Qr(torch.cat([x, soh], dim=1))
        
        return soc, soh, Qr
    
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        init.ones_(m.weight) 
        init.zeros_(m.bias) 
    elif isinstance(m, nn.BatchNorm1d):
        init.ones_(m.weight)  
        init.zeros_(m.bias)  
    elif isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

if __name__=="__main__":
    model = BatteryCHR()
    summary(model, input_size=(32, 4))