from pytorch_lightning import Trainer
from modules import InBreastSystem
import yaml

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
       config = yaml.load(f, Loader=yaml.FullLoader)
    model = InBreastSystem(config)
    trainer = Trainer(max_nb_epochs=20)
    trainer.fit(model)
