import glob
import os

import torch

from config import load_config
from net import GloveModel
from trainer import Trainer
from utils import fix_seed
import time


def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print(
            "WorkingTime[{}]: {} sec".format(
                original_fn.__name__, end_time - start_time
            )
        )
        return result

    return wrapper_fn


@logging_time
def main(hparams):
    fix_seed(hparams.seed)
    scaler = torch.cuda.amp.GradScaler() if hparams.amp else None
    model = GloveModel(1, hparams.emb_dimension)

    # training phase
    trainer = Trainer(hparams, model, scaler)
    version = trainer.fit()


if __name__ == "__main__":
    hparams = load_config()
    main(hparams)
