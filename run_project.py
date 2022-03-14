#!/usr/bin/env python
# coding: utf-8
import sys
import getopt

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

# TODO: [V] Give experiment_module a name
# from experiments.ex1.experiment_module import ExperimentConfig


seed_everything(1, workers=True)
# Load Arguments
try:
    opts, args = getopt.getopt(sys.argv[1:], "m:e:l:g:c:")
except getopt.GetoptError:
    print('run_project.py -e <experiment_name> -m <run_mode> [-l <log_dir>] [-g <gpu_count>] [-c <cpu_count>]')
    sys.exit(2)
experiment_name = dict(opts)['-e']
run_mode = dict(opts)['-m']

if '-l' in dict(opts):
    log_dir = dict(opts)['-l']  # e.g., '/home/ai/work/logs/tensorboard'
else:
    log_dir = 'logs/tensorboard'


if '-g' in dict(opts):
    gpu_count = int(dict(opts)['-g'])  # e.g., 1
else:
    gpu_count = None 

if '-c' in dict(opts):
    cpu_count = int(dict(opts)['-c'])  # e.g., 16
else:
    cpu_count = 4


print("experiment_name:", experiment_name)
print("run_mode:", run_mode)

if run_mode != 'fit1batch' and run_mode != 'fastdebug' and run_mode != 'train' and run_mode != 'test':
    print('run_project.py -e <experiment_name> -m <run_mode> [-l <log_dir>]')
    print('<run_mode> should be "fit1batch", "fastdebug", "train", or "test"')
    print('<log_dir> is the directory to store tensorboard logs. "logs/tensorboard" by default.')
    sys.exit(2)

# Import experiment package
exec(f'from experiments.{experiment_name}.experiment_module import ExperimentConfig')
exec(f'from experiments.{experiment_name}.experiment_module import ExperimentalMultiTaskDataModule')
exec(f'from experiments.{experiment_name}.experiment_module import ExperimentalMultiTaskModule')

# Load experiment objects
datamodule = ExperimentalMultiTaskDataModule(num_workers=cpu_count, pin_memory=(gpu_count is not None)) # 
# - Note: pin_memory = True only when GPU is available, or the program may slowdown dramatically.

datamodule.prepare_data()

print('MultiTaskDataModule Built')

module = ExperimentalMultiTaskModule(ExperimentConfig.experiment_parameters)

print('MultiTaskModule Built')

logger = TensorBoardLogger(log_dir,
                           ExperimentConfig.name,
                           default_hp_metric=False,
                           log_graph=True)

checkpoint = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    dirpath=f'./checkpoint/{ExperimentConfig.name}',
    filename='epoch{epoch:02d}-loss{best_val_loss:.2f}',
    save_last=True,
    auto_insert_metric_name=False,
    every_n_val_epochs=1,
    verbose=True
)

early_stopping = EarlyStopping('val_loss', mode='min', verbose=True)

# lr_monitor = LearningRateMonitor(logging_interval='epoch', log_momentum=True)


# Start Running ...
if run_mode == 'fit1batch':
    # Overfit Test:
    # Note: Why you should always overfit a single batch to debug your deep learning model?
    # https://www.youtube.com/watch?v=nAZdK4codMk
    trainer = pl.Trainer(
        auto_scale_batch_size='power',
        auto_lr_find=True,
        logger=logger,
        callbacks=[checkpoint],
        deterministic=True,
        num_sanity_val_steps=0,
        overfit_batches=1,
        gpus=gpu_count,
        auto_select_gpus=True,
        accelerator="ddp"
    )
    # Notes:
    # num_sanity_val_steps should be set so that
    # 1. model architecture can be plot in the beginning.
    # 2. validation step can be tested before training start.
    trainer.tune(module, datamodule=datamodule)
    
    # datamodule.batch_size = 64
    # module.lr = 2e-2
    trainer.fit(module, datamodule=datamodule)
elif run_mode == 'fastdebug' or run_mode == 'train':
    trainer = pl.Trainer(
        auto_scale_batch_size='power',
        auto_lr_find=True,
        logger=logger,
        callbacks=[checkpoint, early_stopping],
        deterministic=True,
        num_sanity_val_steps=1,  # Debug
        fast_dev_run=(run_mode == 'fastdebug'),
        gpus=gpu_count,
        auto_select_gpus=True,
        accelerator="ddp"
    )
    trainer.tune(module, datamodule=datamodule)
    trainer.fit(module, datamodule=datamodule)
elif run_mode == 'test':
    module = MultiTaskModule.load_from_checkpoint(
        f'checkpoint/{ExperimentConfig.name}/{ExperimentConfig.best_model_checkpoint}',
        parameters=ExperimentConfig.experiment_parameters
    )
    pl.Trainer().test(module, datamodule=datamodule)

# TODO:
# - [V] 要用torch metric 才會快: https://torchmetrics.readthedocs.io/en/latest/?_ga=2.184197729.610530333.1621525374-364715702.1621241882
# - [ ] logging of hyperparameters: https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#logging-hyperparameters
#       - [V] show hparams and metric values on Tensorboard
#       - [ ] logging of trial mode : success, fail, running, unknown
# - [V] Survey and organize the tools in https://pytorch-lightning.readthedocs.io/en/1.2.2/common/trainer.html (找出適合Debug用的工具)
#       - [V] fast_dev_run
#       - [V] overfit_batches
# - [V] Adopt lightning Callbacks :
#       - [V] EarlyStopping: https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.callbacks.EarlyStopping.html#pytorch_lightning.callbacks.EarlyStopping
#       - [V] LearningRateMonitor: https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.callbacks.LearningRateMonitor.html#pytorch_lightning.callbacks.LearningRateMonitor
#       - [V] ModelCheckpoint: https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint
#       - [V] Make sure we can load checkpoint for testing
# - [V] 實作lightning DataModule以進一步把Data的部分和pl_module解偶: https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#why-do-i-need-a-datamodule
# - [V] Identify @ExperimentDependent parts in run_project.py and move them to experiment_module.py.
# - [ ] Organize Folder Structure (share + common)
#       - [V] make sure preprocess.py can be run in the ex1 directory
#       - [ ] Move all 'path' to an .ini file
#       - [V] allow directory of import to be redirected during runtime.
# - [ ] Allow using arbitrary score for checkpoint and early stop (need to self.log in pl_module.py)
# - [ ] In pl_module, need to have a strategy for splitting training into train and 'val', other than using 'test' for 'val'.
# - [ ] Incorporate with ray[tune]. Ref: https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-lightning.html
# - [X] XXX: come up with some thing to do cross-validation. Ref: https://towardsdatascience.com/5x-faster-scikit-learn-parameter-tuning-in-5-lines-of-code-be6bdd21833c. (p.s., Cross validation is often not used for evaluating deep learning models because of the greater computational expense)
# - [ ] Many speed up tips: https://pytorch-lightning.readthedocs.io/en/stable/benchmarking/performance.html


#