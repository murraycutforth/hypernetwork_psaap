import pickle
from typing import List, Tuple
import logging

import pandas as pd
import torch

from src.utils.paths import project_dir

logger = logging.getLogger(__name__)


def tensor_list_from_pressure_traces() -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Load pressure traces from exp and convert to tensor. Also handles normalisation of pressure values,
    and conversion to torch.Tensor with dtype=torch.float32.
    """

    data_filepath = project_dir() / 'data' / 'exp' / 'pressure_change.pkl'

    with open(data_filepath, 'rb') as f:
        data = pickle.load(f)

    test_data_sheet_path = project_dir() / 'data' / 'exp' / 'gp_exp_dataset.csv'
    test_data_sheet = pd.read_csv(test_data_sheet_path, header=0)
    test_data_sheet['test_num'] = test_data_sheet['test_num'].astype(int)
    test_data_sheet['spark_x'] = test_data_sheet['spark_x'].astype(float)
    test_data_sheet['spark_z'] = test_data_sheet['spark_z'].astype(float)
    test_data_sheet['energy'] = test_data_sheet['energy'].astype(float)

    trace_list = []
    xi_list = []

    for k, trace in data.items():

        def get_xi_from_run_number(run_number):
            matching_rows = test_data_sheet[test_data_sheet['test_num'] == run_number]

            if not len(matching_rows) == 1:
                return None

            row = matching_rows.iloc[0]
            xi = torch.tensor([row['spark_x'], row['spark_z'], row['energy']], dtype=torch.float32)
            return xi

        xi = get_xi_from_run_number(k)

        if xi is None:
            logger.warning(f"Run number {k} not found in test data sheet")
            continue

        xi_list.append(xi)

        trace = trace / 25000.0

        assert trace.min() >= -0.2, f"Min: {trace.min()}"
        assert trace.max() <= 2.5, f"Max: {trace.max()}"

        trace = torch.tensor(trace, dtype=torch.float32).view(-1, 1)

        trace_list.append(trace)

    def normalise_xi_list(xi_list):
        xi_list = torch.stack(xi_list)
        xi_mean = xi_list.mean(dim=0)
        xi_std = xi_list.std(dim=0)
        xi_list = (xi_list - xi_mean) / xi_std
        return [x for x in xi_list]

    xi_list = normalise_xi_list(xi_list)

    return trace_list, xi_list
