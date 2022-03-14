#!/usr/bin/env python
# coding: utf-8
import torch.nn.functional as F

from common.utils import blockPrinting
from common.pl_module import BaseMultiTaskModule, BaseMultiTaskDataModule

from experiments.ex4.preprocess.config import PreProcess
from experiments.ex4.model import MultiTaskModel

import os

origin_path = 'data/source'
experiment_name = __file__.split("/")[-2]  # same as the name of current folder
preprocess = PreProcess(save_tmp=True, experiment_name=experiment_name)

preprocess.config(
    chid_file=os.path.join(origin_path, 'sample_chid.txt'),
    cdtx_file=os.path.join(origin_path, 'sample_zip_if_cca_cdtx0001_hist.csv'),
    cust_f_file=os.path.join(origin_path, 'sample_zip_if_cca_cust_f.csv'),
    category_cols=['chid', 'bnsfg', 'iterm', 'mcc', 'scity'],
    sparse_feat=['chid', 'bnsfg', 'iterm', 'mcc', 'scity'],
    numeric_cols=['bnspt', 'timestamp_0', 'timestamp_1', 'objam'],
    dense_feat=['bnspt', 'timestamp_0', 'timestamp_1', 'objam'],
    cust_category_cols=['chid', 'masts', 'educd', 'trdtp', 'poscd'],
    cust_numeric_cols=['slam', 'first_mob', 'constant_change', 'sum_l2_ind',
                       'sum_u2_ind', 'constant_l2_ind', 'constant_u4_ind',
                       'growth_rate', 'monotone_down', 'monotone_up', 'data_dt'],
    target_cols=['chid', 'data_dt', 'objam_sum', 'objam_mean', 'trans_count', 'objam_mean_M3'],
    USE_CHID=True,
    time_column_data_dt='data_dt',
    time_column_csmdt='csmdt',
    result_column_timestamp_1='timestamp_1',
    result_column_timestamp_0='timestamp_0',
    result_column_timestamp='timestamp',
    LEFT='left',
    INNER='inner',
    n_sample=1000,
    window_size=120,
    test_size=2
)


@blockPrinting
def _get_data_dependent_model_parameters(pipe):
    # @DataDependent
    use_chid = pipe.USE_CHID.get()

    dense_dims = pipe.dense_dims.get(load_tmp=True)
    sparse_dims = pipe.sparse_dims.get(load_tmp=True)

    num_y_data = 3

    ground_truths = [pipe.test_objmean, pipe.test_tscnt, pipe.test_label_0]

    out_dims = [y_data.get(load_tmp=True).shape[1] for y_data in ground_truths]

    class_outputs = [type(y_data.get(load_tmp=True)[0].item()) != float for y_data in ground_truths]

    return {
        'dense_dims': dense_dims,
        'sparse_dims': sparse_dims,
        'use_chid': use_chid,
        'out_dims': out_dims,
        'class_outputs': class_outputs
    }


class ExperimentConfig:
    # @ExperimentDependent
    best_model_checkpoint = 'epoch07-loss0.00.ckpt'

    name = experiment_name

    experiment_parameters = {
        "model_parameters": {
            "data_independent": {
                'hidden_dims': 64,
                'n_layers': 2,
                'cell': 'LSTM',
                'bi': False
            },
            "data_dependent": _get_data_dependent_model_parameters(preprocess.pipe)
        },
        "training_parameters": {
            "dropout": 0.5,
            "warmup_epochs": 5,
            "annealing_cycle_epochs": 40
        }
    }


class ExperimentalMultiTaskDataModule(BaseMultiTaskDataModule):
    # TODO:
    # [V] allow batch_size to be saved as parameter

    def __init__(self, batch_size=64, num_workers=4, pin_memory=False):
        super(ExperimentalMultiTaskDataModule, self).__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    # @blockPrinting
    def prepare_data(self):
        # things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode.
        # e.g., download
        self.train_dataset = preprocess.pipe.train_dataset.get()
        self.test_dataset = preprocess.pipe.test_dataset.get()


class ExperimentalMultiTaskModule(BaseMultiTaskModule):

    def __init__(self, experiment_parameters, lr=2e-4):
        super(ExperimentalMultiTaskModule, self).__init__(experiment_parameters, lr=lr)

    def config_model(self, model_parameters, dropout):
        return MultiTaskModel(
            model_parameters,
            dropout=dropout
        )

    def config_loss_funcs(self):
        return [F.mse_loss, F.mse_loss, F.binary_cross_entropy]

    def config_task_names(self):
        return ['objmean', 'tscnt', 'label_0']

    def config_task_metrics(self):
        return {
            'objmean': ['mse', 'mae'],
            'tscnt': ['mse', 'mae'],
            'label_0': ['acc', 'auc']
        }
    '''
    If additional metrics are considered in the experiment,
    we can override the following function with new metric calculator added.
    def config_metric_calculators(self):
        from torchmetrics import MeanSquaredError, MeanAbsoluteError, Accuracy, AUROC
        return {
            'mse': lambda: MeanSquaredError(compute_on_step=False),
            'mae': lambda: MeanAbsoluteError(compute_on_step=False),
            'acc': lambda: Accuracy(compute_on_step=False),
            'auc': lambda: AUROC(compute_on_step=False, pos_label=1)
        }
    '''

    def batch_forward(self, batch):
        # TODO: should be override >>
        # @DataDependent -> input and output definition
        x_dense, x_sparse, objmean, tscnt, label_0 = batch
        outputs = self(x_dense, x_sparse)
        ground_truths = objmean, tscnt, label_0
        return outputs, ground_truths
