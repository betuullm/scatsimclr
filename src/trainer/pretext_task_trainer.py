from typing import NoReturn, Tuple
from pathlib import Path
import shutil

import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.loss import NTXentLoss
from src.data import PretextTaskDatasetWrapper
from src.data import EmbeddingExtractor
from src.evaluation import LogisticRegressionEvaluator
from src.models import ResNetSimCLR, ScatSimCLR
from src.data.datasets import NUM_CLASSES


def get_device() -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def save_config_file(model_checkpoints_folder: Path) -> NoReturn:
    model_checkpoints_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy('./config.yaml', model_checkpoints_folder / 'config.yaml')


class PretextTaskTrainer:

    EMBEDDINGS_MODELS = ['resnet18', 'resnet50', 'scatsimclr8', 'scatsimclr12', 'scatsimclr16', 'scatsimclr30']

    def __init__(self, config):
        self._config = config

        self._jigsaw = config['pretext']['jigsaw']
        self._rotation = config['pretext']['rotation']

        if sum((self._jigsaw, self._rotation)) > 0 or sum((self._jigsaw, self._rotation)) == 0:
            raise ValueError('Only one pretext task should be selected at the time')

        self._device = get_device()
        self._writer = SummaryWriter()

        self._nt_xent_criterion = NTXentLoss(self._device, config['batch_size'], **config['loss'])

        h_size = 2048  # size of the h feature space
        num_jigsaw_permutations = config['pretext']['num_jigsaw']

        if self._jigsaw:
            self._n_jigsaw = 9

            self._fc1 = nn.Sequential(
                # first FC
                nn.Linear(h_size, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5)
            )

            self._fc2 = nn.Sequential(
                # second FC
                nn.Linear(self._n_jigsaw * 512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5)
            )

            self._classification_fc = nn.Sequential(
                nn.Linear(4096, num_jigsaw_permutations)
            )

        if self._rotation:

            self._fc1 = nn.Sequential(
                # first FC
                nn.Linear(h_size, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5)
            )

            self._fc2 = nn.Sequential(
                # second FC
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5)
            )

            self._classification_fc = nn.Sequential(
                nn.Linear(128, 4)
            )

        self._classification_loss = nn.CrossEntropyLoss()
        self._loss_lambda = 0.3

        self._name_postfix = 'jigsaw' if self._jigsaw else 'rotation'

    def train(self) -> NoReturn:
        dataset_wrapper = PretextTaskDatasetWrapper(batch_size=self._config['batch_size'],
                                                    jigsaw=self._config['pretext']['jigsaw'],
                                                    rotation=self._config['pretext']['rotation'],
                                                    **self._config['dataset'])
        train_loader, valid_loader = dataset_wrapper.get_data_loaders()

        # create and if needed load model
        model = self._get_embeddings_model(self._config['model']['base_model'])
        model = self._load_weights(model)

        optimizer = torch.optim.Adam(params=list(model.parameters()) + list(self._fc1) + list(self._fc2) +
                                            list(self._classification_fc),
                                     lr=3e-4, weight_decay=self._config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader),
                                                               eta_min=0, last_epoch=-1)

        # create checkpoint and save
        checkpoint_folder = Path(self._writer.log_dir) / 'checkpoints'
        save_config_file(checkpoint_folder)

        n_iter = 0
        valid_n_iter = 0
        test_n_iter = 0
        best_valid_loss = np.inf
        best_acc = -1

        for epoch_counter in range(1, self._config['epochs'] + 1):

            # run training
            for i, inputs in enumerate(train_loader):
                optimizer.zero_grad()

                loss_contrastive, loss_pretext, acc_pretext = self._step(model, inputs)

                if i % self._config['log_every_n_steps'] == 0:
                    self._log_to_tensorboard(loss_contrastive, loss_pretext, acc_pretext, n_iter, 'train')

                loss = loss_contrastive + self._loss_lambda * loss_pretext
                n_iter += 1

                loss.backward()
                optimizer.step()

            # validation
            if epoch_counter % self._config['validate_every_n_epochs'] == 0:
                loss_contrastive_valid, loss_pretext_valid, acc_pretext_valid = self._validate(model, valid_loader)
                self._log_to_tensorboard(loss_contrastive_valid, loss_pretext_valid, acc_pretext_valid,
                                         valid_n_iter, mode='valid')
                valid_n_iter += 1

                # save the best model
                if best_valid_loss > loss_contrastive_valid:
                    best_valid_loss = loss_contrastive_valid
                    torch.save(model.state_dict(), checkpoint_folder / 'model.pth')
                    torch.save({"fc1": self._fc1.state_dict(), "fc2": self._fc2.state_dict(),
                                "classification_fc": self._classification_fc.state_dict()},
                               checkpoint_folder / f'model_{self._name_postfix}.pth')

            # classification
            if epoch_counter % self._config['eval_every_n_epochs'] == 0:
                acc = self._test_classification(model)

                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), checkpoint_folder / 'model.pth')

                self._writer.add_scalar('test/classification_accuracy', acc, test_n_iter)
                test_n_iter += 1

            # schedule lr
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self._writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

        # save final model
        torch.save(model.state_dict(), checkpoint_folder / 'model_final.pth')
        torch.save({"fc1": self._fc1.state_dict(), "fc2": self._fc2.state_dict(),
                    "classification_fc": self._classification_fc.state_dict()},
                   checkpoint_folder / f'model_{self._name_postfix}_final.pth')

    def _validate(self, model: nn.Module,
                  valid_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, float]:

        with torch.no_grad():
            # freeze weights
            model.eval()
            self._fc1.eval()
            self._fc2.eval()
            self._classification_fc.eval()

            loss_contrastive_total = 0.
            loss_pretext_total = 0.
            acc_pretext_total = 0.
            counter = 0

            for inputs in valid_loader:
                loss_contrastive, loss_pretext, acc_pretext = self._step(model, inputs)

                loss_contrastive_total += loss_contrastive
                loss_pretext_total += loss_pretext
                acc_pretext_total += acc_pretext
                counter += 1

            loss_contrastive_total /= counter
            loss_pretext_total /= counter
            acc_pretext_total /= counter

        # unfreeze weights
        model.train()
        self._fc1.train()
        self._fc2.train()
        self._fc2.train()
        return loss_contrastive, loss_pretext, acc_pretext

    def _test_classification(self, model: nn.Module) -> float:
        """Tests classification with extracted features

        Args:
            model: CLR model to use to compute embeddings for classification

        Returns:
            float: classification accuracy
        """

        model.eval()

        input_size = eval(self._config['dataset']['input_size'])
        dataset = self._config['dataset']['dataset']
        epochs = 100

        extractor = EmbeddingExtractor(model, device=self._device,
                                       dataset=dataset,
                                       input_size=input_size, batch_size=self._config['batch_size'])
        train_data, train_labels, test_data, test_labels = extractor.get_features()

        evaluator = LogisticRegressionEvaluator(n_features=train_data.shape[1],
                                                n_classes=NUM_CLASSES[dataset],
                                                device=self._device, batch_size=64)
        accuracy = evaluator.run_evaluation(test_data, train_labels, test_data, test_labels, epochs)
        return accuracy

    def _log_to_tensorboard(self, loss_contrastive: torch.Tensor,
                            loss_pretext: torch.Tensor,
                            acc_pretext: float,
                            global_count: int,
                            mode: str = 'train') -> NoReturn:
        name = self._name_postfix
        self._writer.add_scalar(f'{mode}/loss_{name}', float(loss_pretext), global_count)
        self._writer.add_scalar(f'{mode}/acc_{name}', float(acc_pretext), global_count)
        self._writer.add_scalar(f'{mode}/loss_contrastive', float(loss_contrastive), global_count)

    def _step(self, model: nn.Module,
              inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:

        img1 = inputs[0].to(self._device)
        img2 = inputs[1].to(self._device)
        input_pretext = inputs[2]
        label_pretext = inputs[3]
        y_ = label_pretext.view(-1).to(self._device)

        _, zis = model(img1)
        _, zjs = model(img2)

        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        loss_contrastive = self._nt_xent_criterion(zis, zjs)

        B, T, C, H, W = input_pretext.size()
        input_pretext = input_pretext.view(B * T, C, H, W).to(self._device)
        pretext_h, _ = model(input_pretext)

        if self._jigsaw:
            pretext_h = pretext_h.view(B, T, -1)
            pretext_h = pretext_h.transpose(0, 1)

            z_list = []
            for i in range(self._n_jigsaw):
                z = self._fc1(pretext_h[i])
                z = z.view([B, 1, -1])
                z_list.append(z)

            h_ = torch.cat(z_list, 1)
            h_ = self._fc2(h_.view(B, -1))

        elif self._rotation:
            h_ = pretext_h.squeeze()
            h_ = self._fc1(h_)
            h_ = self._fc2(h_)

        h_ = self._classification_fc(h_)
        pred = torch.max(h_, 1)
        acc = torch.sum(pred[1] == y_).cpu().numpy() * 1.0 / len(y_)
        return loss_contrastive, self._classification_loss(h_, y_), acc

    def _get_embeddings_model(self, model_name: str) -> nn.Module:

        if model_name not in self.EMBEDDINGS_MODELS:
            raise ValueError('Unsupported model')

        if 'resnet' in model_name:
            return ResNetSimCLR(**self._config['model'])

        if 'scatsimclr' in model_name:
            blocks = int(model_name[10:])
            input_size = eval(self._config['dataset']['input_shape'])

            return ScatSimCLR(**self._config['model'], input_size=input_size, res_blocks=blocks)

    def _load_weights(self, model: nn.Module) -> nn.Module:
        checkpoints_folder = Path('./runs') / f"{self._config['fine_tune_from']}/checkpoints"

        if checkpoints_folder.exists():
            state_dict = torch.load(checkpoints_folder / 'model_final.pth')
            model.load_state_dict(state_dict)
        else:
            print('Pre-trained weights not found. Training from scratch.')
        return model
