import argparse
import importlib.util
from utils.exp import init_experiment
from engine.trainer import Trainer
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from dataset.YasmineDataset2 import YKDataset
from utils.log import logger

VAL_SPLIT = .2
SHUFFLE_DATASET = True
RANDOM_SEED = 8


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_path', type=str,
                    help='Path to the model script.')
    ap.add_argument("-w", "--window_size", default=192, type=int,
                    help='The size of the sliding window.')
    ap.add_argument("-e", "--n_epochs", default=1000, type=int,
                    help='number of epochs.')
    ap.add_argument("-b", "--batch_size", default=16, type=int,
                    help='The batch size.')
    ap.add_argument("-n", "--experiment_name", default='', type=str,
                    help='The name of the experiment.')
    ap.add_argument("-o", "--optimizer", default='adam', type=str,
                    help='The name of the network optimizer.'
                         'possible choice: adam, adamw, sgd')
    ap.add_argument("-l", "--learning_rate", default=0.001, type=float,
                    help='The learning rate of the optimizer.')
    return ap.parse_args()


def main():
    args = parse_args()
    model_script = load_module(args.model_path)
    cfg = init_experiment(args)
    model, model_cfg = model_script.init_model(cfg)

    # prepare dataset
    dataset = YKDataset(window_size=cfg.window_size, dic_path=cfg.DIC_PATH)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(VAL_SPLIT * dataset_size))
    if SHUFFLE_DATASET:
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    trainer = Trainer(model, cfg, model_cfg, dataset, train_sampler, val_sampler, cfg.optimizer)

    logger.info(f'Total Epochs: {cfg.n_epochs}')
    for epoch in range(cfg.n_epochs):
        trainer.training(epoch)
        trainer.validation(epoch)


def load_module(script_path):
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)

    return model_script


if __name__ == "__main__":
    main()
