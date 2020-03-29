from pytorch_lightning import Trainer
from tqdm.auto import tqdm
from modules import INBreastClassification
from callbacks import checkpoint_callback
import yaml


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = INBreastClassification(config)
    trainer = Trainer(max_epochs=20, gpus=1, progress_bar_refresh_rate=1, checkpoint_callback=checkpoint_callback)
    trainer.fit(model)
