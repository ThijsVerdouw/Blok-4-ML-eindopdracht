from mads_datasets.base import BaseDatastreamer
from mltrainer.preprocessors import BasePreprocessor
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch import nn
import torch
from pathlib import Path
from typing import Dict

import ray
import torch
from filelock import FileLock
from loguru import logger
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics, rnn_models
from mltrainer.preprocessors import PaddedPreprocessor
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

import sys
import datasets, metrics
sys.path.append('../')
import models

logger.info('started')

trainfile = Path('../data/heart_train.parq').resolve()
testfile = Path('../data/heart_test.parq').resolve()
# trainfile = Path('../data/heart_big_train.parq').resolve()
# testfile = Path('../data/heart_big_test.parq').resolve()

traindataset = datasets.HeartDataset2D(trainfile, target="target")
testdataset = datasets.HeartDataset2D(testfile, target="target")
traindataset, testdataset
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = "cpu"

traindataset.to(device)
testdataset.to(device)

trainstreamer = BaseDatastreamer(traindataset, preprocessor = BasePreprocessor(), batchsize=32)
teststreamer = BaseDatastreamer(testdataset, preprocessor = BasePreprocessor(), batchsize=32)
f1micro = metrics.F1Score(average='micro')
f1macro = metrics.F1Score(average='macro')
precision = metrics.Precision('micro')
recall = metrics.Recall('macro')
accuracy = metrics.Accuracy()

SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float


def train(config: Dict):
    """
    The train function should receive a config file, which is a Dict
    ray will modify the values inside the config before it is passed to the train
    function.
    """
    
    model = models.CNN(config)
    model.to(device)

    trainersettings = TrainerSettings(
        epochs=10,
        metrics=[accuracy, f1micro, f1macro, precision, recall],
        logdir="heart2D",
        train_steps=len(trainstreamer),
        valid_steps=len(teststreamer),  # type: ignore
        reporttypes=[ReportTypes.RAY],
        scheduler_kwargs={"factor": 0.2, "patience": 5},
        earlystop_kwargs=None,
    )

    # because we set reporttypes=[ReportTypes.RAY]
    # the trainloop wont try to report back to tensorboard,
    # but will report back with ray
    # this way, ray will know whats going on,
    # and can start/pause/stop a loop.
    # This is why we set earlystop_kwargs=None, because we
    # are handing over this control to ray.

    trainer = Trainer(
        model=model,
        settings=trainersettings,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,  # type: ignore
        traindataloader=trainstreamer.stream(),
        validdataloader=teststreamer.stream(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

    trainer.loop()

try:
    ray.init()
except:
    ray.shutdown()
    ray.init()

tune_dir = Path().resolve()
config = {
    # "input_size": 1, irrelevant
    "num_classes": 2,      #5 for big one
    "tune_dir": tune_dir,
    "hidden": 16, # tune.randint(16, 128),
    "dropout": 0.2, # tune.uniform(0.1, 0.4),
    "num_layers": tune.randint(1, 4)
}

reporter = CLIReporter()
reporter.add_metric_column("accuracy")
# reporter.add_metric_column("precision")
# reporter.add_metric_column("recall")

bohb_hyperband = HyperBandForBOHB(
    time_attr="training_iteration",
    max_t=50,
    reduction_factor=3,
    stop_last_trials=False,
)

bohb_search = TuneBOHB()

analysis = tune.run(
    # trainstreamer,
    train,
    config=config,
    metric="test_loss",
    mode="min",
    progress_reporter=reporter,
    local_dir=str(config["tune_dir"]),
    num_samples=20,
    search_alg=bohb_search,
    scheduler=bohb_hyperband,
    verbose=1,
)

ray.shutdown()