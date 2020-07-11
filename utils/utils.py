import os
import sys
import numpy as np
import torch
import random
from loguru import logger
import copy
import json


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_logger(log_dir, to_file=True):
    fmt = "{time} | {level} | {message}"
    if to_file:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_name = os.path.join(log_dir, "runtime_{time}.log")
        logger.add(file_name, format=fmt, level="DEBUG", encoding='utf-8', )
    else:
        logger.remove()
        logger.add(sys.__stdout__, colorize=True, format=fmt, level="INFO")


def config_to_dict(config):

    output = copy.deepcopy(config.__dict__)
    output['device'] = config.device.type
    if hasattr(config, 'embedding_pretrained'):
        output.pop('embedding_pretrained')
    return output


def config_to_json_string(config):
    """Serializes this instance to a JSON string."""
    return json.dumps(config_to_dict(config), indent=2, sort_keys=True, ensure_ascii=False) + '\n'
