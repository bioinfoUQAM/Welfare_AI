from easydict import EasyDict as edict
from models.modelling.LeNet_sigmoid2 import LeNet
from torch import nn


def init_model(cfg):
    model_cfg = edict()
    model_cfg.criterion = nn.BCELoss()

    model = LeNet()
    model.to(cfg.device)

    return model, model_cfg