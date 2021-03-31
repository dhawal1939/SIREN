import torch
from torch.utils.data import Dataset

from kornia.filters import laplacian, spatial_gradient

def generate_coordinates(n: int) -> torch.Tensor:
    '''
    Genearates grid of 2D coordinates [0, n]x[n, 0]
    params
    -----------
    n: int
        Number of 2d points
    returns:
    -----------
    coord_abs: torch.ndarray
        image coordinates of (n**2) x 2 size
    '''
    #meshgrid of torch uses ij so lets just use it 
    #rather than use np meshgrid and change the 
    # indexing to ij
    with torch.no_grad():
        rows, cols = torch.meshgrid(torch.arange(n), torch.arange(n))
        # i , j format coordinates
        coords_abs = torch.stack([torch.ravel(rows), torch.ravel(cols)], axis=-1)
    
    return coords_abs


class pixel_dataset(Dataset):
    '''
    Custom Dataloader for the torch training
    params:
    ------------
    size: int
        height and width of image
    coords_abs: torch.tensor
        image coordinates (n**2) x 2 size
    grad: torch.tensor
        gradient approximation in two directions x, y of size (size x size, 2) 
    grad_norm: torch.tensor
        gradient image(approx) normalized of size (size x size)
    laplace: torch.tensor
        laplace of the image/signal to be precise
    '''
    def __init__(self, img):
        if not img.dim() == 2 or not img.size()[0] == img.size()[1]:
            raise ValueError('Image should be single channel square image')
        with torch.no_grad():
            #creating dataset
            self.img = img
            self.size = img.size()[0]
            self.coords_abs = generate_coordinates(self.size)
            # better not normalize
            self.grad = spatial_gradient(img.view(1, 1, self.size, self.size), mode='sobel', normalized=False)[0][0]
            self.grad = torch.stack((self.grad[1], self.grad[0]), axis=0)
            self.grad_norm = torch.linalg.norm(self.grad, dim=0)
            
            self.grad = self.grad.permute(1, 2, 0)
            self.laplace = laplacian(img.view(1, 1, self.size, self.size), kernel_size=3, normalized=False)[0][0]
        
    def __len__(self):
        '''number of samples :) (pixels :( )'''
        return self.size ** 2
    
    def __getitem__(self, idx):
        '''get all relavant data for one single coordinate'''
        with torch.no_grad():
            coords_abs = self.coords_abs[idx]
            r, c = coords_abs
            
            coords = 2 * ((coords_abs / self.size) - .5) # change scale
            
        return {
                    'coords': coords,
                    'coords_abs': coords_abs,
                    'intensity': self.img[r, c],
                    'grad_norm': self.grad_norm[r, c],
                    'grad': self.grad[r, c],
                    'laplace': self.laplace[r, c]
               }
        