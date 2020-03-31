import pytorch_lightning as pl
from losses import f1_precis_recall
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

checkpoint_callback = ModelCheckpoint(
    filepath='./checkpoints/Recall-{epoch}-{val_los:.2f}-{F1:.2f}-{recall:.2f}',
    save_top_k=2,
    verbose=True,
    monitor='recall'
)

f1_checkpoint = ModelCheckpoint(
    filepath='./checkpoints/F1_Score-{epoch}-{val_los:.2f}-{F1:.2f}-{recall:.2f}',
    save_top_k=2,
    verbose=True,
    monitor='F1'
)
