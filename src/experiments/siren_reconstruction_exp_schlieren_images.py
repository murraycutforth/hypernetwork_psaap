"""
In this script, we evaluate the ability of SIREN models to reconstruct individual schlieren images from LF simulations

"""
import json
import logging
from pathlib import Path
from typing import List
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.metrics import compute_all_metrics
from src.utils.paths import project_dir
from src.models.siren import Siren

logger = logging.getLogger(__name__)



def main():
    results_path = project_dir() / 'output' / 'reconstruction_exp_results.json'
    tensor_list = tensor_list_from_image()

    lrs = [1.25e-4, 5e-4, 4e-3]
    img_inds = [0, 5, 10, 15, 20, 24]
    hidden_features = [256]

    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = []
        for lr, ind, fts in itertools.product(lrs, img_inds, hidden_features):
            results.append(train_eval_siren(tensor_list, ind=ind, lr=lr, hidden_features=fts))

        with open(results_path, 'w') as f:
            f.write(json.dumps(results, indent=4))

    df = results_to_dataframe(results)
    plot_results_df(df)



def train_eval_siren(img_list: list,
                     ind: int = 0,
                     hidden_features: int = 256,
                     hidden_layers: int = 3,
                     img_sidelength: int = 256,
                     loss_atol: float = 1e-6,
                     lr: float = 1e-4,
                     max_steps: int = 250
                     ):
    """Train a SIREN model to reconstruct a single image and evaluate its performance.
    """
    params = {
        'img_sidelength': img_sidelength,
        'hidden_features': hidden_features,
        'hidden_layers': hidden_layers,
        'loss_atol': loss_atol,
        'lr': lr,
        'max_steps': max_steps,
        'ind': ind
    }

    logging.info(f'Calling train_eval_siren with params: ind={ind}, img_sidelength={img_sidelength}, hidden_features={hidden_features}, hidden_layers={hidden_layers}, loss_atol={loss_atol}, lr={lr}')

    model = Siren(in_features=2, out_features=1, hidden_features=hidden_features, hidden_layers=hidden_layers, outermost_linear=True)
    # Number of parameters in model
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters in model: {num_params}")
    dataloader = DataLoader(ImageFittingDataset(img_sidelength, img_list[ind]), batch_size=1, pin_memory=True, num_workers=0)
    model_input, ground_truth = next(iter(dataloader))

    model.train()

    loss_vals = []

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    while True:
        model_output, coords = model(model_input)
        loss = ((model_output - ground_truth) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        loss_vals.append(loss.item())
        optim.step()

        if check_atol_criterion(loss_vals, loss_atol) or len(loss_vals) > max_steps:
            break

    plot_loss(loss_vals)
    plot_final_prediction(model_output, ground_truth, img_sidelength)

    metrics = final_eval_siren(model, model_input, ground_truth)
    metrics['steps'] = len(loss_vals)

    return {'params': params, 'metrics': metrics}


def plot_results_df(df):
    """Plot the results, using different image runs to compute variance in metrics
    """
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for ind in df['ind'].unique():
        df_sub = df[df['ind'] == ind]
        color = plt.cm.viridis(ind / df['ind'].max())
        ax[0].scatter(df_sub['hidden_features'], df_sub['mse'], label=f'ind={ind}', color=color)
        ax[1].scatter(df_sub['hidden_features'], 1 - df_sub['ssim'], label=f'ind={ind}', color=color)
        ax[2].scatter(df_sub['hidden_features'], df_sub['psnr'], label=f'ind={ind}', color=color)

    ax[0].set_title('MSE')
    ax[1].set_title('1 - SSIM')
    ax[2].set_title('PSNR')

    for x in ax:
        x.set_xscale('log')
        x.set_xlabel('Learning Rate')
        x.set_yscale('log')

    fig.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for ind in df['ind'].unique():
        df_sub = df[df['ind'] == ind]
        color = plt.cm.viridis(ind / df['ind'].max())
        ax[0].scatter(df_sub['lr'], df_sub['mse'], label=f'ind={ind}', color=color)
        ax[1].scatter(df_sub['lr'], 1 - df_sub['ssim'], label=f'ind={ind}', color=color)
        ax[2].scatter(df_sub['lr'], df_sub['psnr'], label=f'ind={ind}', color=color)

    ax[0].set_title('MSE')
    ax[1].set_title('1 - SSIM')
    ax[2].set_title('PSNR')

    for x in ax:
        x.set_xscale('log')
        x.set_xlabel('Learning Rate')
        x.set_yscale('log')

    fig.tight_layout()
    plt.show()
    plt.close()


def results_to_dataframe(results):
    df = pd.DataFrame([{'img_sidelength': r['params']['img_sidelength'],
                        'hidden_features': r['params']['hidden_features'],
                        'hidden_layers': r['params']['hidden_layers'],
                        'loss_atol': r['params']['loss_atol'],
                        'lr': r['params']['lr'],
                        'ind': r['params']['ind'],
                        'max_steps': r['params']['max_steps'],
                        'mse': r['metrics']['mse'],
                        'ssim': r['metrics']['ssim'],
                        'psnr': r['metrics']['psnr'],
                        'steps': r['metrics']['steps']} for r in results])
    return df


def check_atol_criterion(loss_vals, atol, n=10):
    """Check if the loss has reached the specified absolute tolerance for each of the past n steps
    """
    if len(loss_vals) < n + 1:
        return False

    return all(abs(x - y) < atol for x, y in zip(loss_vals[-n:], loss_vals[-n-1:]))


def plot_final_prediction(model_output, ground_truth, img_sidelength):
    """Plot the final model prediction and the ground truth image
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(model_output.detach().numpy().squeeze().reshape(img_sidelength, img_sidelength))
    ax[0].set_title('Predicted')
    ax[1].imshow(ground_truth.detach().numpy().squeeze().reshape(img_sidelength, img_sidelength))
    ax[1].set_title('Ground Truth')
    plt.show()
    plt.close()



def final_eval_siren(model: Siren, model_input: torch.Tensor, ground_truth: torch.Tensor):
    """Evaluate a trained SIREN model on a single image.
    """
    model.eval()
    model_output, coords = model(model_input)
    metrics = compute_all_metrics(model_output.detach().numpy().squeeze(),
                                  ground_truth.detach().numpy().squeeze())
    logger.info(f"Final metrics: {metrics}")

    return metrics


def plot_loss(loss_vals):
    """Plot the loss curve after training a model
    """
    plt.plot(loss_vals)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.show()
    plt.close()


def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x, y, z, ...) coordinates in a range of -1 to 1.
    """
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class ImageFittingDataset(Dataset):
    """This dataset class is used to fit a SIREN model to a single image.
    """
    def __init__(self, sidelength, img):
        super().__init__()
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError
        return self.coords, self.pixels






def tensor_list_from_image(sidelength: int = 256) -> List[torch.Tensor]:
    """Load schlieren images from exp and convert to tensor
    """

    data_dir = project_dir() / 'data' / 'exp' / 'images'

    run_dirs = sorted(list(data_dir.glob('*')))

    run_dir = run_dirs[0]  # Only one run for now

    print(run_dir)

    file_paths = sorted(list(run_dir.glob('*.png')))

    x = list(PIL.Image.open(x) for x in file_paths)
    x = list(np.array(im) / 255 for im in x)
    x = list(PIL.Image.fromarray(im) for im in x)

    transform = Compose([
        Resize((sidelength, sidelength)),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))  # Normalize to [-1, 1]
    ])

    imgs = list(map(transform, x))

    return imgs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s : %(message)s")
    main()