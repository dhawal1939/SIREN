import torch


class gradient_utils:
    @staticmethod
    def gradient(target, coords):
        '''
        Compute the gradient with respect to input
        -------------
        params
        -------------
        target: torch.Tensor
            2D tensor of shape (n_coords, ?) representing the targets
        -------------
        coords: torch.Tensor
            2D tensor of shape (n_coords, 2) representing the coordinates
        -------------
        returns: grad: torch.Tensor
            2D tensor of shape (n_coords, 2) representing the gradient
        '''
        return torch.autograd.grad(
                                        target,
                                        coords,
                                        grad_outputs=torch.ones_like(target),
                                        create_graph=True,
                                  )[0]
    
    @staticmethod
    def divergence(grad, coords):
        '''
        Compute the second order partial derivative when input is gradient
        2D case it is F_{xx} + F_{yy}
        --------------
        params
        --------------
        grad : torch.Tensor
            2D tensor of shape (n_corrds, 2) representing the gradient w.r.t x and y
        --------------
        coords: torch.Tensor
            2D tensor of shape (n_coords, 2) representing the coordinates
        ---------------
        returns: divergence: torch.Tensor
            2D tensor of shape (n_coords, 1) representing the divergence
        '''
        div = .0
        for i in range(coords.shape[1]):
            div += torch.autograd.grad(
                                            grad[..., i],
                                            coords,
                                            torch.ones_like(grad[..., i]),
                                            create_graph=True,
                                      )[0][..., i: i + 1]
        return div
        
    @staticmethod
    def laplace(target, coords):
        '''
        Computer Laplace
        ------------------
        params
        ------------------
        target: torch.Tensor
            2D tensor of shape (n_coords, 1) representing the targets
        ------------------
        coords: torch.Tensor
            2D tensor of shape (n_coords, 2) representing the coordinates
        ------------------
        returns: laplace: torch.Tensor
            2D tensor of shape (n_coords, 1) representing the laplace
         '''
        grad = gradient_utils.gradient(target, coords)
        
        return gradient_utils.divergence(grad, coords)
        