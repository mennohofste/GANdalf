#!/home/menno/Documents/GANdalf/.venv/bin/python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from network.architecture import CycleGAN
from network.architecture import StarGAN
from data import FaceDataModule


def main():
    checkpoint_callback = ModelCheckpoint(
        monitor='val/l1',
        dirpath='checkpoints',
        filename='StarGAN-epoch{epoch:02d}-l1{val/l1:.2f}',
        auto_insert_metric_name=False,
        save_last=True,
        save_top_k=3
    )

    model = StarGAN()
    trainer = Trainer(callbacks=[checkpoint_callback],
                      gpus=1, max_epochs=10)
    dm = FaceDataModule()

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
