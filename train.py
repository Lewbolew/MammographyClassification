from pytorch_lightning import Trainer
from tqdm.auto import tqdm
from modules import INBreastClassification
from callbacks import f1_checkpoint, early_stopping, recall_checkpoint
import yaml

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = INBreastClassification(config)

    trainer = Trainer(max_epochs=150, gpus=[1], progress_bar_refresh_rate=1, checkpoint_callback=f1_checkpoint, nb_sanity_val_steps=0)
    trainer.fit(model)