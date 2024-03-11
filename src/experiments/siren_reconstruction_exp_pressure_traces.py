"""
In this script, we evaluate the ability of SIREN models to reconstruct individual schlieren images from LF simulations

"""
import json
import logging
import itertools

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.data import tensor_list_from_pressure_traces
from src.utils.metrics import compute_all_metrics
from src.utils.paths import project_dir
from src.models.siren import Siren

logger = logging.getLogger(__name__)



def main():
    tensor_list = tensor_list_from_pressure_traces()
    results_path = project_dir() / 'output' / 'reconstruction_exp_pressure_results.json'

    lrs = [5e-4]
    hidden_features = [4, 8, 16, 32]
    inds = list(range(len(tensor_list)))

    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = []
        for lr, ind, fts in itertools.product(lrs, inds, hidden_features):
            results.append(train_eval_siren(tensor_list, ind=ind, lr=lr, hidden_features=fts))

        with open(results_path, 'w') as f:
            f.write(json.dumps(results, indent=4))

    df = results_to_dataframe(results)
    plot_results_df(df)



def train_eval_siren(pressure_list: list,
                     ind: int = 0,
                     hidden_features: int = 64,
                     hidden_layers: int = 3,
                     loss_atol: float = 1e-12,
                     lr: float = 1e-4,
                     max_steps: int = 2500
                     ):
    """Train a SIREN model to reconstruct a single pressure trace and evaluate its performance.
    """
    params = {
        'hidden_features': hidden_features,
        'hidden_layers': hidden_layers,
        'loss_atol': loss_atol,
        'lr': lr,
        'max_steps': max_steps,
        'ind': ind
    }

    logging.info(f'Calling train_eval_siren with params: hidden_features={hidden_features}, hidden_layers={hidden_layers}, loss_atol={loss_atol}, lr={lr}')

    model = Siren(in_features=1, out_features=1, hidden_features=hidden_features, hidden_layers=hidden_layers, outermost_linear=True)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters in model: {num_params}")

    dataloader = DataLoader(PressureFittingDataset(pressure_list[ind]), batch_size=1, pin_memory=True, num_workers=0)
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
    plot_final_prediction(model_output, ground_truth)

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
        x.set_xlabel('Hidden Features')
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
    df = pd.DataFrame([{'hidden_features': r['params']['hidden_features'],
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


def plot_final_prediction(model_output, ground_truth):
    """Plot the final model prediction and the ground truth image
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(model_output.detach().numpy().squeeze().reshape(25))
    ax[0].set_title('Predicted')
    ax[1].plot(ground_truth.detach().numpy().squeeze().reshape(25))
    ax[1].set_title('Ground Truth')
    plt.show()
    plt.close()



def final_eval_siren(model: Siren, model_input: torch.Tensor, ground_truth: torch.Tensor):
    """Evaluate a trained SIREN model on a single image.
    """
    model.eval()
    model_output, coords = model(model_input)
    metrics = compute_all_metrics(model_output.detach().numpy().squeeze(),
                                  ground_truth.detach().numpy().squeeze(),
                                  data_range=3)
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


class PressureFittingDataset(Dataset):
    """This dataset class is used to fit a SIREN model to a single image.
    """
    def __init__(self, pressure_trace):
        super().__init__()
        self.pressure_trace = pressure_trace
        self.coords = torch.linspace(0, 1, 25).reshape(-1, 1)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError
        return self.coords, self.pressure_trace


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s : %(message)s")
    main()