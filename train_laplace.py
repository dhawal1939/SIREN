import os, sys
import math
import pathlib
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

from model import *
from gradient_utils import *
from utils import *


img_path  = pathlib.Path.cwd() / 'imgs' / 'shaderball.png'
img_ = read_image(str(img_path), ImageReadMode.GRAY)/ 255.
img = 2 * (img_ - .5)
img = torch.squeeze(img)
downsampling_factor = 2
img =  img[::downsampling_factor, ::downsampling_factor]
dataset = pixel_dataset(torch.squeeze(img))

#Hyperparameters
epochs = 15000
batch_size = img.size()[1] ** 2
log_freq = 3

hidden_features = 256
hidden_layers = 3


siren_model = image_siren(
                            hidden_layers=hidden_layers,
                            hidden_features= hidden_features,
                            hidden_omega=30
                         )


print('Sin Representation Network', siren_model)

data_loader = DataLoader(dataset, batch_size=batch_size)

optim_siren = torch.optim.Adam(lr=1e-4, params=siren_model.parameters())

siren_model = siren_model.cuda() if torch.cuda.is_available() else siren_model

siren_model.train()

siren_criterion = nn.MSELoss()

siren_criterion = siren_criterion.cuda() if torch.cuda.is_available() else siren_criterion

for epoch in range(epochs):
    relu_losses, siren_losses = [], []
    with tqdm(total=len(data_loader), desc=f'Epoch {epoch} Trained Batches') as progress:
        for i, batch in enumerate(data_loader):
            x = batch['coords'].to(torch.float32)
            x.requires_grad = True
            x = x.cuda() if torch.cuda.is_available() else x

            #Predicts the image
            y_pred_siren = siren_model(x)

            '''
            Optimize on all
            '''

            '''
            Intensity Loss
            '''

            # y_true = batch['intensity'][:, None].to(torch.float32)
            # y_true = y_true.cuda() if torch.cuda.is_available() else y_true

            # intensity_loss = siren_criterion(y_pred_siren, y_true)
            # del y_true

            '''
            Gradient Loss
            '''
            # y_true_grad = batch['grad'].to(torch.float32)
            # y_true_grad = y_true_grad.cuda() if torch.cuda.is_available() else y_true_grad

            # y_pred_grad_siren = gradient_utils.gradient(y_pred_siren, x)

            # gradient_loss = siren_criterion(y_pred_grad_siren, y_true_grad)
            # del y_true_grad
            
            '''
            Laplace Loss
            '''
            y_true_laplace = batch['laplace'][:, None].to(torch.float32)
            y_true_laplace = y_true_laplace.cuda() if torch.cuda.is_available() else y_true_laplace

            y_pred_laplace_siren = gradient_utils.laplace(y_pred_siren, x)

            laplace_loss = siren_criterion(y_pred_laplace_siren, y_true_laplace)
            del y_true_laplace

            '''
            Combined Loss
            '''
            siren_loss = laplace_loss

            siren_losses.append(siren_loss.item())

            optim_siren.zero_grad()
            siren_loss.backward()
            optim_siren.step()

            progress.set_postfix_str(
                                        s=f'SIREN loss {torch.mean(torch.Tensor(siren_losses))}',
                                        refresh=True
                                    )
            progress.update()
            
    if epoch % log_freq == 0:
        import numpy as np
        pred_img = np.zeros_like(img)
        pred_img_grad_norm = np.zeros_like(img)
        pred_img_laplace = np.zeros_like(img)

        orig_img = np.zeros_like(img)
        
        with tqdm(total=len(data_loader), desc=f'Logging') as progress:
            for d_batch in data_loader:
                coords = d_batch["coords"].to(torch.float32)
                coords.requires_grad = True
                coords_abs = d_batch["coords_abs"].numpy()
                
                coords = coords.cuda() if torch.cuda.is_available() else coords
                
                pred = siren_model(coords)
                pred_n = pred.detach().cpu().numpy().squeeze()
                pred_grad_norm = (
                                    gradient_utils.gradient(pred, coords)
                                    .norm(dim=-1)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .squeeze()
                                 )
                pred_laplace = gradient_utils.laplace(pred, coords).detach().cpu().numpy().squeeze()

                pred_img[coords_abs[:, 0], coords_abs[:, 1]] = pred_n
                pred_img_grad_norm[coords_abs[:, 0], coords_abs[:, 1]] = pred_grad_norm
                pred_img_laplace[coords_abs[:, 0], coords_abs[:, 1]] = pred_laplace
                
                progress.update()

            fig, axs = plt.subplots(3, 2, constrained_layout=True)
            axs[0, 0].imshow(dataset.img, cmap="gray")
            axs[0, 1].imshow(pred_img, cmap="gray")

            axs[1, 0].imshow(dataset.grad_norm, cmap="gray")
            axs[1, 1].imshow(pred_img_grad_norm, cmap="gray")

            axs[2, 0].imshow(dataset.laplace, cmap="gray")
            axs[2, 1].imshow(pred_img_laplace, cmap="gray")

            for row in axs:
                for ax in row:
                    ax.set_axis_off()

            fig.suptitle(f"Iteration: {epoch}")
            axs[0, 0].set_title("Ground truth")
            axs[0, 1].set_title("Prediction")

            plt.savefig(f"visualization_laplace/{epoch}.png")
