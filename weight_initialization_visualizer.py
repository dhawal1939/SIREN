import pathlib
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *


torch.manual_seed(2)

### VARIOUS INIT FUNCTIONS ###
init_functions = {
                    'ones':torch.nn.init.ones_,
                    'eye':torch.nn.init.eye_,
                    'default': partial(torch.nn.init.kaiming_uniform_, a=5 ** (.5)),
                    'paper': None
                 }

for init_name, init_function in init_functions.items():
    path = pathlib.Path.cwd() / 'tensorboard_logs' / init_name
    writer = SummaryWriter(log_dir=path)
    
    def layer_logger(inst, inp, out, number=0):
        layer_name = f"{number}_{inst.__class__.__name__}"
        writer.add_histogram(layer_name, out)
    
    model = image_siren(
                            hidden_layers=10,
                            hidden_features=200,
                            first_omega=30,
                            hidden_omega=30,
                            c=6,
                            custom_initalizationg_function=init_function
                        )
    model = model.cuda() if torch.cuda.is_available() else model
    
    for i, layer in enumerate(model.model.modules()):
        if not i:
            continue
        layer.register_forward_hook(partial(layer_logger, number=(i + 1) // 2))
    
    inp = 2 * (torch.rand(10000, 2) - .5)
    
    inp = inp.cuda() if torch.cuda.is_available() else inp
    
    writer.add_histogram('0', inp)
    
    res = model(inp)
    
    del model, inp
    torch.cuda.empty_cache()