import multiprocessing
from lightfm import LightFM
from pysmore import SMORe
import numpy as np
import pandas as pd
from datasets import RecDataset, Interaction, User, Item
from evaluation import Evaluation

class RecTrainer:

    def __init__(self, 
                 model,
                 model_type,
                 model_path,
                 dataset, 
                 evaluation=None,
                 model_name=None,
                 compute_metrics='P@5'):

        self.model = model
        self.model_name = model_name
        self.model_type = model_type
        self.model_path = model_path
        self.dataset = dataset
        self.evaluation = Evaluation(dataset_instance=dataset) if evaluation is None else evaluation
        self.metrics = compute_metrics if isinstance(compute_metrics, list) else [compute_metrics]

    def save_model(self):
        pass

    def run_smore(self, update_times=10, is_training=True):
        if is_training:
            self.model.fit(update_times=update_times)
        else:
            n_user, n_item = self.dataset.matrices['eval']['interactions'].shape
            result_matrix = self.model.predict()
            return result_matrix.reshape(n_user, n_item)

    def run_lightfm(self, is_training=True):
        """
        Function tailored for lightfm, which the training input is the csr interaction matrices, 
        get the training x type, includes (1) cf (id) (2) meta (feature) (3) hybid (id+feature),
        then training for an epoch only. Call 'trainer.train()' or 'trainer.eval()'.

        [TBC] Cold-start issue: 
            (1) remove cold-start in training 
            (2) evaluate warm-start then cold-start

        Note: 
            - Parameter of lightfm: 
                no_components (latent dimensions)
                k (for k-OS loss), 
                n (maximum positive), 
                max_sampled (WARP loss sampling)
            - cold-start process: (these two use meta feature)
                1. Used the original dummy (see dataset.cold_start)
                2. Trained a dummy for all the common user/item, 
                    and replace the cold-start used/item's embeddings
        """
        if self.model_type == 'hybrid':
            x = 'ids_and_features'
        elif self.model_type == 'meta':
            x = 'features'
        elif self.model_type == 'cf':
            x = 'ids'
        else:
            exit(0)

        if is_training:
            self.model.fit_partial(
                    interactions=self.dataset.matrices['train']['interactions'],
                    user_features=self.dataset.matrices['train'][f'user_{x}'],
                    item_features=self.dataset.matrices['train'][f'item_{x}'],
                    epochs=1, 
            )
        else:
            n_user, n_item = self.dataset.matrices['eval']['interactions'].shape

            result_matrix = self.model.predict(
                    user_ids=np.repeat(np.arange(n_user), n_item),
                    item_ids=np.tile(np.arange(n_item), n_user),
                    user_features=self.dataset.matrices['train'][f'user_{x}'],
                    item_features=self.dataset.matrices['train'][f'item_{x}'],
                    num_threads=multiprocessing.cpu_count()
            )
            while np.any(result_matrix==0):
                print('Smooth the result with non-zero by adding one.')
                results_matrix += 1
            return result_matrix.reshape(n_user, n_item)


    def train(self,
              epochs=None,
              do_eval=False,
              model_type="",
              verbose=False,
              eval_per_epochs=1):
        """
        Function for training fm for n epochs, includes the evaluation each epochs

        Args:
            epochs: training epochs, i.e. training the whole dataset n times.
            do_eval: besides training, conduct the evaluation for the corresponding predictions.
            verbose: training detail logging.
            eval_per_epochs: evaluation at the given training steps (epochs).
        """
        for epoch in range(epochs):
            # lightFM
            if verbose:
                print(f"***** Epoch: {epoch+1} *****")

            # smore has only one epoch
            if isinstance(self.model, SMORe):
                self.run_smore(update_times=epochs)

                # [DEBUG] one-shot training for SMORe,
                if do_eval:
                    score_matrix = self.run_smore(is_training=False)
                    self.evaluation.eval(
                            prediction=score_matrix, 
                            metrics=self.metrics,
                            exclude_cold_start=False,
                            max_positive_truth=None,
                            do_heuristic=False 
                    )
                    results = self.evaluation.get_average_eval(
                            steps=f'SMORe {model_type} [{epoch+1}]', 
                            do_heuristic=False
                    )
                    print(f"{results}\n")

                break

            # lightfm
            if isinstance(self.model, LightFM):
                # training each epoch
                self.run_lightfm()

                # evaluate each specified epoch
                if do_eval and ( (epoch+1) % eval_per_epochs == 0 or epoch == epochs - 1):
                    score_matrix = self.run_lightfm(is_training=False)
                    self.evaluation.eval(
                            prediction=score_matrix, 
                            metrics=self.metrics,
                            exclude_cold_start=False,
                            max_positive_truth=None,
                            do_heuristic=True if epoch == epochs - 1 else False
                    )
                    results = self.evaluation.get_average_eval(
                            steps=f'LightFM {model_type} [{epoch+1}]', 
                            do_heuristic=True if epoch == epochs - 1 else False
                    )
                    print(f"{results}\n")

        return self.evaluation

    def evaluate(self, user_ids, item_ids):
        """
        [TBA] Function for evaluate the given user item pairs.
        """
        pass
    
    def __repr__(self):
        NULL=None
        return f"RecTrainer({{\
                \n\tmodel: {self.model_name}\
                \n\tmodel type: {self.model_type}\
                \n\tmodel path: {self.model_path}\
                \n\taverage_subscribing_times: {NULL}\
                \n}})"
