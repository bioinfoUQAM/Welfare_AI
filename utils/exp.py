import yaml
from easydict import EasyDict as edict
from pathlib import Path
import torch
from .log import logger, add_logging
import pprint


def init_experiment(args):
    # add path from config.yml to cfg
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        cfg = edict(cfg)

    # add args to cfg
    update_cfg(cfg, args)

    # prepare experiment repository
    experiments_path = Path(cfg.EXPS_PATH)
    last_exp_indx = find_last_exp_indx(experiments_path)
    experiment_name = f'{last_exp_indx:03d}'
    if cfg.experiment_name:
        experiment_name += '_' + cfg.experiment_name
    exp_path = experiments_path / experiment_name
    exp_path.mkdir(parents=True)
    cfg.EXP_PATH = exp_path

    # add other elements to cfg
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg.CHECKPOINTS_PATH = exp_path / 'checkpoints'
    cfg.LOGS_PATH = exp_path / 'logs'
    cfg.CHECKPOINTS_PATH.mkdir(exist_ok=True)
    cfg.LOGS_PATH.mkdir(exist_ok=True)

    # add logs
    add_logging(cfg.LOGS_PATH, prefix='train_')
    logger.info('Run experiment with config:')
    logger.info(pprint.pformat(cfg, indent=4))

    return cfg


def update_cfg(cfg, args):
    for param_name, value in vars(args).items():
        if param_name.lower() in cfg or param_name.upper() in cfg:
            continue
        cfg[param_name] = value


def find_last_exp_indx(exp_parent_path):
    indx = 0
    for x in exp_parent_path.iterdir():
        if not x.is_dir():
            continue

        exp_name = x.stem
        if exp_name[:3].isnumeric():
            indx = max(indx, int(exp_name[:3]) + 1)

    return indx
