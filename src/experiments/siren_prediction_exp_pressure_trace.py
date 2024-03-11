"""In this script we apply a SIREN-based hypernetwork to predict the pressure trace from experiment
"""

import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt

from src.utils.data import tensor_list_from_pressure_traces
from src.models.meta_modules import PressureTraceHypernet
from src.utils.paths import project_dir
#from src.utils.training import train

logger = logging.getLogger(__name__)


def main():
    # Construct train and test dataloaders
    train_ds, test_ds = train_val_test_split()
    train_dl = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    logger.debug(f"Number of training samples: {len(train_ds)}")
    logger.debug(f"Number of test samples: {len(test_ds)}")
    logger.debug(f"Number of batches in training dataloader: {len(train_dl)}")
    logger.debug(f"Number of batches in test dataloader: {len(test_dl)}")
    shape_dict = {k: v.shape for k, v in next(iter(train_dl)).items()}
    logger.debug(f"First batch in train dataloader: {shape_dict}")



    # Construct hypernetwork-based model
    # The hypo_model is a SIREN model which maps from time to pressure
    # The hyper_model is a hypernetwork which maps from a \\xi value to the parameters of the hypo_model
    model = PressureTraceHypernet()

    # Test forward pass
    model_input = next(iter(train_dl))
    model_output = model(model_input)

    # Test backwards pass
    loss = (model_output['model_out'] - model_output['model_in']['p'])**2
    loss = loss.mean()
    loss.backward()
    logger.debug(f"\n\n\nBackwards pass computed. Loss: {loss}")

    name_to_none_grads = {name: param.grad for name, param in model.named_parameters() if param.grad is None}
    logger.debug(f"None grads: {name_to_none_grads}")

    name_to_grads = {name: param.grad.shape for name, param in model.named_parameters() if param.grad is not None}
    logger.debug(f"Grads: {name_to_grads}")

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-6)
    outdir = project_dir() / 'siren_hypernetwork'

    max_epochs = 5000
    avg_train_losses, avg_val_losses = [], []
    avg_param_vals, std_param_vals = [], []
    avg_param_vals_hypernet, std_param_vals_hypernet = [], []

    visualise_val_preds_and_gt(model, test_dl, step=0)

    for epoch in range(max_epochs):

        model.train()
        train_losses = []
        for batch in train_dl:
            optimizer.zero_grad()
            model_output = model(batch)
            loss = (model_output['model_out'] - model_output['model_in']['p'])**2
            loss = loss.mean()
            loss.backward()

            # This is necessary, particularly with multi-layer hyponetwork
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)

            # TODO: the average hypo-net params are only meaningful if we have a single run in the dataset
            avg_param_vals.append([param.abs().mean().item() for k, param in model_output['hypo_params'].items()])
            std_param_vals.append([param.abs().std().item() for k, param in model_output['hypo_params'].items()])
            avg_param_vals_hypernet.append([param.abs().mean().item() for param in model.hyper_net.parameters()])
            std_param_vals_hypernet.append([param.abs().std().item() for param in model.hyper_net.parameters()])

            optimizer.step()
            train_losses.append(loss.item())

        avg_train_losses.append(np.mean(train_losses))
        logger.info(f"Train loss: {np.mean(train_losses)}")

        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch in test_dl:
                model_output = model(batch)
                loss = (model_output['model_out'] - model_output['model_in']['p'])**2
                loss = loss.mean()
                val_losses.append(loss.item())

        avg_val_losses.append(np.mean(val_losses))
        logger.info(f"Test loss: {np.mean(val_losses)}")

        if epoch % 100 == 0:
            visualise_val_preds_and_gt(model, test_dl, step=epoch)

        logger.info(f"Epoch {epoch} completed")

    # Plot losses
    plt.plot(avg_train_losses, label='Train loss')
    plt.plot(avg_val_losses, label='Test loss')
    plt.legend()
    plt.yscale('log')
    plt.show()

    # Plot average parameter values
    plt.title("Average parameter values in hypo-network")
    plt.plot([np.mean(x) for x in avg_param_vals], label='Average parameter value')
    plt.plot([np.nanmean(x) for x in std_param_vals], label='Std parameter value')
    plt.plot([np.mean(x) for x in avg_param_vals_hypernet], label='Average parameter value hypernet')
    plt.plot([np.nanmean(x) for x in std_param_vals_hypernet], label='Std parameter value hypernet')
    plt.legend()
    plt.show()



def visualise_val_preds_and_gt(model, test_dl, step):
    """Visualise the predictions and ground truth for the validation set
    """
    n_rows = len(test_dl) // 8 + 1
    n_cols = 8

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), sharex=True)
    fig.suptitle(f"Validation set predictions at step {step}")

    model.eval()
    with torch.no_grad():
        for k, batch in enumerate(test_dl):
            model_output = model(batch)

            i = k // n_cols
            j = k % n_cols

            ax = axs[i, j]
            ax.plot(model_output['model_in']['t'].numpy().squeeze(), model_output['model_in']['p'].numpy().squeeze(), label='Ground truth')
            ax.plot(model_output['model_in']['t'].numpy().squeeze(), model_output['model_out'].numpy().squeeze(), label='Prediction')
            #ax.set_title(f'$\\xi$ = {model_output["model_in"]["xi"].numpy().squeeze()}')
            ax.set_ylim(-0.2, 2.5)
            ax.axis('off')

    for ax in axs.flatten():
        ax.axis('off')

    fig.tight_layout()
    plt.show()


def train_val_test_split():
    """Split the runs into train, validation, and test sets
    """
    ds = PressureTraceDataset()
    N = len(ds)

    test_ds, train_ds = random_split(PressureTraceDataset(),
                                     [int(0.3*N), N - int(0.3*N)],
                                     generator=torch.Generator().manual_seed(42))

    return train_ds, test_ds


class PressureTraceDataset(Dataset):
    """This dataset class contains (\\xi, t, p) tuples from pressure traces

    Each sample from this class (train or test) will contain a number of values from a single run

    TODO: for now, we will just return an entire trace as a single sample

    """
    def __init__(self):
        self.tensor_list, self.xi_list = tensor_list_from_pressure_traces()

        # TODO:
        self.tensor_list = self.tensor_list
        self.xi_list = self.xi_list

        self.tensor_list = [t.view(-1, 1) for t in self.tensor_list]
        self.t = torch.linspace(0, 1, len(self.tensor_list[0]), dtype=torch.float32).view(-1, 1)

    def __getitem__(self, idx):
        return {'xi': self.xi_list[idx], 't': self.t, 'p': self.tensor_list[idx]}

    def __len__(self):
        return len(self.tensor_list)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()