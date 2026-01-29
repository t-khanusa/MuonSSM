# from mamba_ssm.modules.muonmamba import MuonMamba, MuonMambaConfig
from mamba_ssm.modules.longhorn import MuonLonghornStack, MuonLonghornStackConfig
# from mamba_ssm.modules.muon_mamba import MuonMamba, MuonMambaConfig
# from mambapy.mamba import Mamba, MambaConfig
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer    
import math



class MMA(nn.Module):

    def __init__(self, d_model=32, d_state=64, num_classes=32, input_channels=6, n_layers=2,
                 momentum_beta=0.8, momentum_alpha=1.0, use_newton_schulz=True):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state  # N - SSM state dimension
        self.num_classes = num_classes
        self.n_layers = n_layers # Lưu lại n_layers
        self.use_newton_schulz = use_newton_schulz
                
        # Store momentum parameters
        self.momentum_beta = momentum_beta
        self.momentum_alpha = momentum_alpha
        
        # Initial Conv1d layer: (B,6,L) -> (B,d_model,L) with BatchNorm1d and ReLU
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # Dropout after conv1d
        self.dropout1 = nn.Dropout(0.1)
        
        # MuonLonghorn configuration with Newton-Schulz option
        self.config = MuonLonghornStackConfig(
            d_model=d_model,
            d_state=d_state,  # N - SSM state dimension (user can set d_state=64)
            n_layers=n_layers,
            beta=momentum_beta,
            alpha=momentum_alpha,
            use_newton_schulz=use_newton_schulz
        )
        self.mamba_momentum = MuonLonghornStack(self.config)
        # Dropout before classifier
        self.dropout2 = nn.Dropout(0.1)
        
        # Final linear layer: D -> n_class
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Initialize conv1d layers
        self.conv1d.apply(self._init_weights)
        
        # Initialize classifier with depth-scaled std (helps with training stability)
        std = self.config.base_std / math.sqrt(2 * self.n_layers)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=std)
        nn.init.zeros_(self.classifier.bias)
        
    def _init_weights(self, module):

        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
        elif isinstance(module, (nn.BatchNorm1d)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):

        # Input: (B, 6, L)
        # Conv1d + BatchNorm1d + ReLU
        x = self.conv1d(x)  # (B, 6, L) -> (B, d_model, L)
        
        x = self.dropout1(x)        
        
        # Transpose for Mamba Momentum: (B, d_model, L) -> (B, L, d_model)
        x = x.transpose(1, 2)
        
        # Mamba
        x = self.mamba_momentum(x)
        
        # Dropout
        x = self.dropout2(x)
        
        # Global Average Pooling: (B, L, d_model) -> (B, d_model)
        x = x.mean(dim=1)
        
        # Classifier
        x = self.classifier(x)
       
        return x



def get_model(args, device = "cuda:0"):
    name = args.name    
    num_classes = args.num_classes
    seq_len = 512
    input_channels = args.input_channels
    n_layers = args.n_layers
    momentum_beta = args.momentum_beta
    momentum_alpha = args.momentum_alpha
    d_model = args.d_model
    d_state = args.d_state
    use_newton_schulz = getattr(args, 'use_newton_schulz', True)  # Default to True for Muon
    print(f"d_model: {d_model}, d_state: {d_state}, num_classes: {num_classes}, input_channels: {input_channels}, n_layers: {n_layers}, momentum_beta: {momentum_beta}, momentum_alpha: {momentum_alpha}, use_newton_schulz: {use_newton_schulz}")

    model = MMA(d_model = d_model, d_state = d_state, num_classes = num_classes, input_channels = input_channels, n_layers = n_layers, momentum_beta = momentum_beta, momentum_alpha = momentum_alpha, use_newton_schulz = use_newton_schulz).to("cuda")
    return model



# input = torch.randn(1, 6, 512).to("cuda:0")
# model = get_model("args", device = "cuda:0")
# print(model)
# output = model(input)
# print(output.shape)