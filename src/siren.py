from pathlib import Path

import PIL
import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid



def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img







class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        #img = get_cameraman_tensor(sidelength)
        img = get_schlieren_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels


if __name__ == "__main__":
    img_dataset = ImageFitting(256)

    dataloader = DataLoader(img_dataset, batch_size=1, pin_memory=True, num_workers=0)

    img_siren = Siren(in_features=2, out_features=1, hidden_features=256,
                      hidden_layers=3, outermost_linear=True)

    total_steps = 500  # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    model_input, ground_truth = next(iter(dataloader))
    #model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    for step in range(total_steps):
        model_output, coords = img_siren(model_input)
        loss = ((model_output - ground_truth) ** 2).mean()

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            img_grad = gradient(model_output, coords)
            img_laplacian = laplace(model_output, coords)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(model_output.cpu().view(256, 256).detach().numpy())
            axes[1].imshow(img_grad.norm(dim=-1).cpu().view(256, 256).detach().numpy())
            axes[2].imshow(img_laplacian.cpu().view(256, 256).detach().numpy())
            plt.show()

        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        coords = get_mgrid(2 ** 10, 1) * 5 * np.pi

        sin_1 = torch.sin(coords)
        sin_2 = torch.sin(coords * 2)
        sum = sin_1 + sin_2

        fig, ax = plt.subplots(figsize=(16, 2))
        ax.plot(coords, sum)
        ax.plot(coords, sin_1)
        ax.plot(coords, sin_2)
        plt.title("Rational multiple")
        plt.show()

        sin_1 = torch.sin(coords)
        sin_2 = torch.sin(coords * np.pi)
        sum = sin_1 + sin_2

        fig, ax = plt.subplots(figsize=(16, 2))
        ax.plot(coords, sum)
        ax.plot(coords, sin_1)
        ax.plot(coords, sin_2)
        plt.title("Pseudo-irrational multiple")
        plt.show()

    with torch.no_grad():
        out_of_range_coords = get_mgrid(1024, 2) * 50
        model_out, _ = img_siren(out_of_range_coords)

        fig, ax = plt.subplots(figsize=(16, 16))
        ax.imshow(model_out.cpu().view(1024, 1024).numpy())
        plt.show()

