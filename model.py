'''
SIREN needs a specific way of initialization for the
Layers
#code inspired from youtuber mildlyoverfitted
'''


import math
import torch
from torch import nn

def initializer(
                    weight: torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor,
                    is_first: bool = True,
                    omega: float = 1.,
                    c: float = 6.
               ) -> None:
    '''
    Intializes the wrights of linear layer
    Weights are based on the paper(supplimentary page 5) presented by SIREN, choose the distribution in range
    For first layer:
        uniform distribution in the range: (-1, 1):
    For rest of layers:
        bound = sqrt(c / (features_in * (omega**2)))
        uniform distribution in the range: [-bound, bound]
        
    params:
            weight: torch.FloatTensor
            ----------------> Initialzing weight for Linear layer
            is_first: bool
            ----------------> If true, the layer is first layer in the network
            omega: float
            ----------------> hyperparameter
            c: float
            ---------------->hyperparameter
    '''
    
    with torch.no_grad():
        bound = 1 / weight.size()[1] if is_first else math.sqrt(c / weight.size()[1]) / omega
        weight.uniform_(-bound, bound)
    return

class siren_layer(nn.Module):
    '''
    SIREN LAYER - Sinusoidal Representation Network Layer
    params:
        in_features: int
            #input_features
        ------------------
        out_features: int
            #out_features
        ------------------
        bias: bool
            True->Needs Bias ; False->Do not Need Bias
        ------------------
        is_first: bool
            True->Is the fisr layers; False->Not First Layer
            Has influence on the initialization of the layer
        ------------------
        omega: flaot
            Hyperparameter helpful in initialization
        ------------------
        c: int
            Hyperparameter helful in initialization
        ------------------
        custom_initalizationg_function: function
            Send custom initialization function accepting a Tensor for initializing it        
    '''
    def __init__(
                    self, 
                    in_features: int,
                    out_features: int,
                    bias: bool = True,
                    is_first: bool = False,
                    omega: float = 30,
                    c: float = 6,
                    custom_initalizationg_function: callable = None
                ):
        
        super(siren_layer, self).__init__()
        
        self.omega, self.c = omega, c
        
        self.linear = nn.Linear(
                                    in_features = in_features,
                                    out_features = out_features,
                                    bias = bias
                                )
        initializer(self.linear.weight, is_first, omega, c) if custom_initalizationg_function is None else custom_initalizationg_function(self.linear.weight)
    
    def forward(self, x):
        '''
        sin( omega * Linear(input))
        '''
        ############################################
        #           LOOK AT THIS LATER             #
        #         Paper says omega.Wx +b           #
        #     while the official implementation    #
        #        does the omega*(Wx + b)           #
        ############################################
        self.linear = self.linear.cuda() if torch.cuda.is_available() else self.linear
#         print(self.linear.weight.size(), x.size(), x.t().size(), self.linear.bias.size())
        return torch.sin(self.omega  * torch.mm(x, self.linear.weight.t()) + self.linear.bias)


class image_siren(nn.Module):
    
    '''
    Image SIREN
    Has a Sequential module inside the class
    params:
        hidden_layers: int
            #hidden_layers
        ------------------
        hidden_features: int
            #hidden_features
        ------------------
        first_omega: flaot
            Hyperparameter helpful in initialization
        ------------------
        hidden_omega: flaot
            Hyperparameter helpful in initialization
        ------------------
        c: int
            Hyperparameter helful in initialization
        ------------------
        custom_initalizationg_function: function
            Send custom initialization function accepting a Tensor for initializing it        
    '''
    def __init__(
                    self,
                    hidden_layers: int = 1,
                    hidden_features: int = 10,
                    first_omega: float = 30,
                    hidden_omega: float = 30,
                    c: float = 6,
                    custom_initalizationg_function: callable = None
              ):
        
        super(image_siren, self).__init__()

        in_features, out_features = 2, 1
        network = []
        
        network.append(
                            # First layer
                            siren_layer(
                                            in_features, #Image coordinates
                                            hidden_features,
                                            is_first=True, # For a different Initialization
                                            custom_initalizationg_function=custom_initalizationg_function,
                                            omega=first_omega,
                                            c=c,
                                            bias=True
                                       )
                      )
        for _ in range(hidden_layers):
            network.append(
                            # Hidden Layes
                            siren_layer(
                                            hidden_features, #Hidden Features
                                            hidden_features,
                                            is_first=False, # For a different Initialization
                                            custom_initalizationg_function=custom_initalizationg_function,
                                            omega=hidden_omega,
                                            c=c,
                                            bias=True
                                       )
                      )
    
        #FINAL LAYER
        network.append( 
                            nn.Linear(hidden_features, out_features)
                      )
        
        initializer(network[-1].weight, False, hidden_omega, c) if custom_initalizationg_function is None else custom_initalizationg_function(network[-1].weight)

        # MODEL
        self.model = nn.Sequential(*network)
    
    def forward(self, x):
        return self.model(x)       
