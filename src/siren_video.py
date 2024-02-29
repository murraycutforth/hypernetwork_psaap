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

def get_mgrid_with_time(sidelen, num_timesteps: int, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    mgrid = mgrid.repeat(num_timesteps, 1)  # Shape is (num_timesteps * sidelen^2, dim)

    time_points = torch.linspace(0, 1, num_timesteps)
    time_points = time_points.repeat_interleave(sidelen ** 2).view(-1, 1)

    mgrid = torch.cat([mgrid, time_points], dim=1)

    return mgrid


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

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


def get_schlieren_tensor(sidelength):
    DATA_DIR = Path.home() / "Downloads" / "schlieren_COARSE_sim_99"
    assert DATA_DIR.exists()
    def load_np_stack():
        files = sorted(DATA_DIR.glob("*.npz"))
        frames = [np.load(f)["F_yz"] for f in files]
        return np.stack(frames, axis=0)

    xs = load_np_stack()

    xs = xs / xs.max()

    transform = Compose([
        Resize((sidelength, sidelength)),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])

    return torch.stack([transform(PIL.Image.fromarray(x)) for x in xs])


class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_schlieren_tensor(sidelength)  # Shape is (25, 1, 128, 128)

        self.pixels = img.permute(0, 2, 3, 1).view(-1, 1)
        self.coords = get_mgrid_with_time(sidelength, num_timesteps=len(img), dim=2)


    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels


def train_mlp(h: int, res: int):
    img_siren = MLP(hidden_features=h)

    total_steps = 500  # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10
    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())
    model_input, ground_truth = next(iter(dataloader))
    # model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
    for step in range(total_steps):
        model_output, coords = img_siren(model_input)
        loss = ((model_output - ground_truth) ** 2).mean()

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))

            fig, axes = plt.subplots(5, 5, figsize=(18, 18))

            for i in range(25):
                axes.flatten()[i].imshow(model_output.cpu().view(25, res, res).detach().numpy()[i])
                axes.flatten()[i].axis('off')
                axes.flatten()[i].set_xticks([])
                axes.flatten()[i].set_yticks([])

            fig.suptitle(f"Step {step}, Total loss {loss.item()}", fontsize=16)
            fig.tight_layout()
            fig.savefig(f"mlp_{res}_{h}_{step:03d}.png")
            plt.close(fig)

        optim.zero_grad()
        loss.backward()
        optim.step()


def train_siren(h: int, res: int):
    img_siren = Siren(in_features=3, out_features=1, hidden_features=h,
                      hidden_layers=3, outermost_linear=True)

    total_steps = 500  # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10
    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())
    model_input, ground_truth = next(iter(dataloader))
    # model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
    for step in range(total_steps):
        model_output, coords = img_siren(model_input)
        loss = ((model_output - ground_truth) ** 2).mean()

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))

            fig, axes = plt.subplots(5, 5, figsize=(18, 18))

            for i in range(25):
                axes.flatten()[i].imshow(model_output.cpu().view(25, res, res).detach().numpy()[i])
                axes.flatten()[i].axis('off')
                axes.flatten()[i].set_xticks([])
                axes.flatten()[i].set_yticks([])

            fig.suptitle(f"Step {step}, Total loss {loss.item()}", fontsize=16)
            fig.tight_layout()
            fig.savefig(f"siren_{res}_{h}_{step:03d}.png")
            plt.close(fig)

        optim.zero_grad()
        loss.backward()
        optim.step()


class MLP(nn.Module):
    def __init__(self, hidden_features):
        super().__init__()
        h = hidden_features
        self.net = nn.Sequential(
            nn.Linear(3, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, coords):
        #coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords


if __name__ == "__main__":
    res = 128
    img_dataset = ImageFitting(res)
    dataloader = DataLoader(img_dataset, batch_size=1, pin_memory=True, num_workers=0)

    for hidden_size in [32, 64, 128, 256]:
        train_siren(hidden_size, res)
        train_mlp(hidden_size, res)


