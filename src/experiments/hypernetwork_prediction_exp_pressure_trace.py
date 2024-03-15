"""In this script we apply a SIREN-based hypernetwork to predict the pressure trace from experiment
"""

import json
import logging
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.data import tensor_list_from_pressure_traces
from src.models.meta_modules import PressureTraceHypernet
from src.utils.paths import project_dir
#from src.utils.training import train

logger = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


def main():
    # Construct train and test dataloaders
    train_ds, test_ds = train_val_test_split()

    # Note: the training dataloader is set such that we do full-batch gradient descent
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
    model = PressureTraceHypernet(hyper_nonlinearity='elu', hypo_nonlinearity='elu')

    with open('model.txt', 'w') as f:
        f.write(str(model))

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
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    outdir = project_dir() / 'siren_hypernetwork'

    max_epochs = 1000
    output_step = 100
    avg_train_losses, avg_val_losses = [], []
    avg_param_vals, std_param_vals = [], []
    avg_param_vals_hypernet, std_param_vals_hypernet = [], []
    all_input_activations, all_output_activations = {}, {}

    visualise_val_preds_and_gt(model, test_dl, step=0)

    for epoch in range(max_epochs):

        # This section of code is devoted to extracting activations before and after each layer (nn.Module)
        # ===========================================================================================

        input_activations_this_epoch = []
        output_activations_this_epoch = []
        input_activations = {}  # Placeholder, this is used by the forward hook
        output_activations = {}  # Placeholder, this is used by the forward hook

        def get_activations(name):
            def hook(model, input, output):
                assert len(input) == 1, input
                input = input[0]

                if isinstance(input, torch.Tensor):
                    logger.debug(f"Input shape: {input.shape} for layer {name}")
                    input_activations[name] = input.detach().numpy().flatten().copy()

                if isinstance(output, torch.Tensor):
                    output_activations[name] = output.detach().numpy().flatten().copy()
            return hook

        if epoch % output_step == 0:
            for name, layer in model.named_modules():
                layer.register_forward_hook(get_activations(name))
        else:
            for name, layer in model.named_modules():
                layer._forward_hooks.clear()

        # ===========================================================================================


        model.train()
        train_losses = []

        for batch in train_dl:
            optimizer.zero_grad()
            model_output = model(batch)

            # Store activations
            input_activations_this_epoch.append(copy.deepcopy(input_activations))
            output_activations_this_epoch.append(copy.deepcopy(output_activations))

            # Compute loss and backprop
            loss = (model_output['model_out'] - model_output['model_in']['p'])**2
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # This is necessary, particularly with multi-layer hyponetwork
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)

            # TODO: the average hypo-net params are only meaningful if we have a single run in the dataset
            avg_param_vals.append([param.abs().mean().item() for k, param in model_output['hypo_params'].items()])
            std_param_vals.append([param.abs().std().item() for k, param in model_output['hypo_params'].items()])
            avg_param_vals_hypernet.append([param.abs().mean().item() for param in model.hyper_net.parameters()])
            std_param_vals_hypernet.append([param.abs().std().item() for param in model.hyper_net.parameters()])

        lr_scheduler.step()

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

                # Store activations
                input_activations_this_epoch.append(copy.deepcopy(input_activations))
                output_activations_this_epoch.append(copy.deepcopy(output_activations))


        avg_val_losses.append(np.mean(val_losses))
        logger.info(f"Test loss: {np.mean(val_losses)}")

        if epoch % output_step == 0:
            input_activations_this_epoch, output_activations_this_epoch = reorganise_activations_this_epoch(input_activations_this_epoch, output_activations_this_epoch)
            all_input_activations[epoch] = input_activations_this_epoch
            all_output_activations[epoch] = output_activations_this_epoch
            #visualise_activations_this_epoch(input_activations_this_epoch, output_activations_this_epoch, epoch)
            #visualise_val_preds_and_gt(model, test_dl, step=epoch)
            #plot_xi_vs_mse(test_dl, model)

        logger.info(f"Epoch {epoch} completed")

    # Final plots
    visualise_val_preds_and_gt(model, test_dl, step=epoch)
    plot_xi_vs_mse(test_dl, model)
    visualise_activations_over_epochs(all_input_activations, all_output_activations)

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


def plot_xi_vs_mse(test_dl, model):
    """Plot the xi values against the mean squared error
    """
    model.eval()
    with torch.no_grad():
        mse_list = []
        xi_list = []
        for batch in test_dl:
            model_output = model(batch)
            mse = (model_output['model_out'] - model_output['model_in']['p'])**2
            mse = mse.mean()
            mse_list.append(mse)
            xi_list.append(model_output['model_in']['xi'].numpy().squeeze())

    mse_list = np.array(mse_list)
    xi_list = np.array(xi_list)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.scatter(xi_list[:, 0], xi_list[:, 1], c=mse_list, cmap='jet')
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('Spark x')
    ax.set_ylabel('Spark z')
    ax.set_title('Mean squared error vs. laser position in val set')

    plt.show()


def reorganise_activations_this_epoch(input_activations_this_epoch, output_activations_this_epoch):
    # First reorganise the activations into a dictionary of lists
    input_activations = {}
    output_activations = {}

    # Check if all the keys are the same
    assert all([set(d.keys()) == set(input_activations_this_epoch[0].keys()) for d in input_activations_this_epoch])
    assert all([set(d.keys()) == set(output_activations_this_epoch[0].keys()) for d in output_activations_this_epoch])

    # Print out difference in keys between input and output activations
    input_keys = set(input_activations_this_epoch[0].keys())
    output_keys = set(output_activations_this_epoch[0].keys())
    print(f'Keys in input activations: {input_keys}')
    print(f'Keys in output activations: {output_keys}')
    print(f'Difference in keys between input and output activations: {input_keys.symmetric_difference(output_keys)}')

    # Delete the keys that are not in both

    for k in input_keys.symmetric_difference(output_keys):
        if k in input_keys:
            input_keys.remove(k)
        else:
            output_keys.remove(k)

    for k in input_keys:
        input_activations[k] = []
        output_activations[k] = []

    for d in input_activations_this_epoch:
        for k in input_keys:
            v = d[k]
            input_activations[k].append(v)

    for d in output_activations_this_epoch:
        for k in output_keys:
            v = d[k]
            output_activations[k].append(v)

    # Flatten the lists

    for k, v in input_activations.items():
        input_activations[k] = np.concatenate(v)
        print(f'Input activations shape for {k}: {input_activations[k].shape}')

    for k, v in output_activations.items():
        output_activations[k] = np.concatenate(v)
        print(f'Output activations shape for {k}: {output_activations[k].shape}')

    # These are dicts mapping from module name to a flat numpy array of all activations for one epoch
    return input_activations, output_activations


def plot_activations_inner(names, all_input_activations, all_output_activations, title):
    assert len(all_input_activations) == len(all_output_activations)
    assert all_input_activations.keys() == all_output_activations.keys()

    for n in names:
        for k in all_input_activations:
            assert n in all_input_activations[k], f"{n} not in input activations"
            assert n in all_output_activations[k], f"{n} not in output activations"

    num_rows = len(names)

    fig, all_axs = plt.subplots(num_rows, 2, figsize=(6, 1 * num_rows), sharex=True)

    num_epochs = len(all_input_activations)
    cmap = plt.get_cmap('cool')
    colors = [cmap(i) for i in np.linspace(0, 1, num_epochs)]

    axs = all_axs[:, 0]
    for i, name in enumerate(names):
        ax = axs[i]
        for k, epoch in enumerate(all_input_activations):
            sns.kdeplot(all_input_activations[epoch][name], color=colors[k], ax=ax, linewidth=0.5)
        ax.set_ylabel(name, rotation=45)
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(-1, 3)
        ax.set_yscale('log')
        ax.set_ylim(0.001, 10)

    axs = all_axs[:, 1]
    for i, name in enumerate(names):
        ax = axs[i]
        for k, epoch in enumerate(all_output_activations):
            sns.kdeplot(all_output_activations[epoch][name], color=colors[k], ax=ax, linewidth=0.5)
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(-1, 3)
        ax.set_yscale('log')
        ax.set_ylim(0.01, 10)

    all_axs[-1, 0].set_xlabel('Input activations')
    all_axs[-1, 1].set_xlabel('Output activations')

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    plt.close()


def visualise_activations_over_epochs(all_input_activations, all_output_activations):
    """Visualise the activations before and after the nonlinearity
    """
    names = ['hyper_net.nets.0.net.0', 'hyper_net.nets.0.net.1', 'hyper_net.nets.0.net.2']
    plot_activations_inner(names, all_input_activations, all_output_activations, title="Activations of hyper network module 0")

    names = [f'hyper_net.nets.{i}' for i in range(10)]
    plot_activations_inner(names, all_input_activations, all_output_activations, title="Activations of hyper network")

    names = ['hypo_net.net.0', 'hypo_net.net.1', 'hypo_net.net.2', 'hypo_net.net.3', 'hypo_net.net.4']
    plot_activations_inner(names, all_input_activations, all_output_activations, title="Activations of hypo network")


def visualise_activations_this_epoch(input_activations, output_activations, epoch):
    """Visualise the activations before and after the nonlinearity
    """
    names = ['hyper_net.nets.0.net.0', 'hyper_net.nets.0.net.1', 'hyper_net.nets.0.net.2']
    plot_activations_inner_one_epoch(names, input_activations, output_activations, title=f"Activations of hyper network module 0 at epoch {epoch}")

    names = [f'hyper_net.nets.{i}' for i in range(10)]
    plot_activations_inner_one_epoch(names, input_activations, output_activations, title=f"Activations of hyper network at epoch {epoch}")

    names = ['hypo_net.net.0', 'hypo_net.net.1', 'hypo_net.net.2', 'hypo_net.net.3', 'hypo_net.net.4']
    plot_activations_inner_one_epoch(names, input_activations, output_activations, title=f"Activations of hypo network at epoch {epoch}")


def plot_activations_inner_one_epoch(names, input_activations, output_activations, title):
    for n in names:
        assert n in input_activations, f"{n} not in input activations"
        assert n in output_activations, f"{n} not in output activations"

    num_rows = len(names)

    fig, all_axs = plt.subplots(num_rows, 2, figsize=(6, 1 * num_rows), sharex=True)

    axs = all_axs[:, 0]
    for i, name in enumerate(names):
        ax = axs[i]
        ax.hist(input_activations[name], bins=100, density=True)
        ax.set_ylabel(name, rotation=45)
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(-1, 1)

    axs = all_axs[:, 1]
    for i, name in enumerate(names):
        ax = axs[i]
        ax.hist(output_activations[name], bins=100, density=True)
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(-1, 3)

    all_axs[-1, 0].set_xlabel('Input activations')
    all_axs[-1, 1].set_xlabel('Output activations')

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    plt.close()


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