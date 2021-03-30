import torch
import torch.nn as nn


class mlp_with_relu(nn.Module):
    '''
    MLP with Relu activation
    ----------
    params
    ----------
    in_features: int
    out_features: int
    hidden_layers: int
    hidden_features: int
    '''
    def __init__(
                    self,
                    in_features: int,
                    out_features: int,
                    hidden_layers: int,
                    hidden_features: int
                 ):
        super(mlp_with_relu, self).__init__()
        
        model_list = [nn.Linear(in_features, hidden_features), nn.ReLU()]
        for _ in range(hidden_layers):
            model_list += [nn.Linear(hidden_features, hidden_features), nn.ReLU()]
        
        model_list.append(nn.Linear(hidden_features, out_features))
        
        self.model = nn.Sequential( *model_list )
        for module in self.model.modules():
            if not isinstance(module, nn.Linear):
                continue
            torch.nn.init.xavier_normal_(module.weight)
    
    def forward(self, x):
        return self.model(x)