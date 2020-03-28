import pytorch_lightning as pl
from losses import f1_precis_recall
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    filepath='./checkpoints/model.ckpt',
    save_top_k=20,
    verbose=True,
    monitor='loss'
)
