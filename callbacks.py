import pytorch_lightning as pl
from losses import f1_precis_recall
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np


early_stopping = EarlyStopping("val_loss")

recall_checkpoint = ModelCheckpoint(
    filepath='./checkpoints/Recall-checkpoint',
    save_top_k=2,
    verbose=True,
    monitor='Recall/val'
)

f1_checkpoint = ModelCheckpoint(
    filepath='./checkpoints/F1_Score-checkpoint.',
    save_top_k=2,
    verbose=True,
    monitor='F1/val',
    mode="max"
)
