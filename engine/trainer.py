import torch
import logging
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
from utils.log import logger, TqdmToLogger
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class Trainer(object):
    def __init__(self, model, cfg, model_cfg,
                 dataset, train_sampler, val_sampler,
                 optimizer='adam',
                 checkpoint_interval=100,
                 save_loss=True, save_accuracy=True):
        self.model = model
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.train_data = DataLoader(dataset, batch_size=cfg.batch_size, sampler=train_sampler)
        self.val_data = DataLoader(dataset, batch_size=cfg.batch_size, sampler=val_sampler)
        if optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        elif optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
        elif optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        else:
            raise NotImplementedError

        self.criterion = self.model_cfg.criterion

        self.device = cfg.device
        self.net = model.to(self.device)

        self.checkpoint_interval = checkpoint_interval

        self.save_loss = save_loss
        if save_loss:
            self.train_loss_file = cfg.LOGS_PATH / 'train_losses.csv'
            with open(self.train_loss_file, 'w') as file:
                file.write("epoch, train_loss\n")

            self.val_loss_file = cfg.LOGS_PATH / 'val_losses.csv'
            with open(self.val_loss_file, 'w') as file:
                file.write("epoch, val_loss\n")

        self.save_accuracy = save_accuracy
        if save_accuracy:
            self.train_accuracy_file = cfg.LOGS_PATH / 'train_accuracy.csv'
            with open(self.train_accuracy_file, 'w') as file:
                file.write("epoch, train_accuracy\n")

            self.val_accuracy_file = cfg.LOGS_PATH / 'val_accuracy.csv'
            with open(self.val_accuracy_file, 'w') as file:
                file.write("epoch, val_accuracy\n")

        # logger
        logger.info(model)
        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

    def training(self, epoch):

        tbar = tqdm(self.train_data, file=self.tqdm_out, ncols=100)

        preds = []
        targets = []
        train_loss = 0.0
        num_batches = 0
        self.net.train()
        for i, (input_data, labels, _) in enumerate(tbar):

            outputs, loss = self.batch_forward(input_data, labels, training=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            train_loss += batch_loss

            num_batches += 1

            tbar.set_description(f'Epoch {epoch}, training loss {train_loss / (i + 1):.6f}')

            if epoch % self.checkpoint_interval == 0:
                self.save_checkpoint(epoch=epoch)

            if epoch == self.cfg.n_epochs-1:
                self.save_checkpoint(epoch=None)

            # accuracy
            predictions = torch.round(outputs)
            preds.append(predictions.data)
            targets.append(labels.data)

        if self.save_loss:
            with open(self.train_loss_file, 'a') as file:
                file.write(f"{epoch}, {train_loss / num_batches:.6f}\n")

        preds = torch.cat(preds)
        targets = torch.cat(targets)
        # print(
        #     f"Accuracy on training set in epoch {epoch} is {accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())}")
        # print(confusion_matrix(targets.cpu().numpy(), preds.cpu().numpy()))
        # print(classification_report(targets.cpu().numpy(), preds.cpu().numpy()))

        if self.save_accuracy:
            with open(self.train_accuracy_file, 'a') as file:
                file.write(f"{epoch}, {accuracy_score(targets.cpu().numpy(), preds.cpu().numpy()):.6f}\n")

    def validation(self, epoch):
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100)

        val_loss = 0
        num_batches = 0
        self.net.eval()
        preds = []
        targets = []
        for i, (input_data, labels, _) in enumerate(tbar):

            outputs, loss = self.batch_forward(input_data, labels, training=False)
            batch_loss = loss.item()
            val_loss += batch_loss

            num_batches += 1

            tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss / (i + 1):.6f}')

            # accuracy
            predictions = torch.round(outputs)
            preds.append(predictions.data)
            targets.append(labels.data)

        if self.save_loss:
            with open(self.val_loss_file, 'a') as file:
                file.write(f"{epoch}, {val_loss / num_batches:.6f}\n")

        preds = torch.cat(preds)
        targets = torch.cat(targets)
        # print(
        #     f"Accuracy on validation set in epoch {epoch} is {accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())}")
        # print(confusion_matrix(targets.cpu().numpy(), preds.cpu().numpy()))
        # print(classification_report(targets.cpu().numpy(), preds.cpu().numpy()))
        # n_correct += (predictions == labels).sum().item()

        if self.save_accuracy:
            with open(self.val_accuracy_file, 'a') as file:
                file.write(f"{epoch}, {accuracy_score(targets.cpu().numpy(), preds.cpu().numpy()):.6f}\n")

    def batch_forward(self, input_data, labels, training=True):
        input_data = input_data.reshape(-1, 8, 4, self.cfg.window_size).to(self.cfg.device)
        labels = labels.to(self.cfg.device)
        # forward
        outputs = self.net(input_data.float())
        loss = self.criterion(outputs, labels)

        return outputs, loss

    def save_checkpoint(self, epoch=None, verbose=False):
        if epoch is None:
            checkpoint_name = 'last_checkpoint.pth'
        else:
            checkpoint_name = f'{epoch:03d}.pth'

        checkpoints_path = self.cfg.CHECKPOINTS_PATH

        if not checkpoints_path.exists():
            self.cfg.CHECKPOINTS_PATH.mkdir(parents=True)

        checkpoint_path = checkpoints_path / checkpoint_name
        if verbose:
            logger.info(f'Save checkpoint to {str(checkpoint_path)}')

        state_dict = self.net.state_dict()
        torch.save(state_dict, str(checkpoint_path))
