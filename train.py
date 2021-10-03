#!/home/menno/Documents/GANdalf/.venv/bin/python
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from network.architecture import CycleGAN
from network.architecture import StarGAN
from network.architecture import StarGAN_2disc
from data import FaceDataModule


def main():
    parser = ArgumentParser()
    parser = CycleGAN.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        monitor='val/lpips',
        filename='CycleGAN-epoch{epoch:02d}-lpips{val/lpips:.2f}',
        auto_insert_metric_name=False,
        save_top_k=3,
    )

    model = CycleGAN(args)
    trainer = Trainer.from_argparse_args(args, callbacks=[checkpoint_callback],
                                         gpus=1, max_epochs=2)
    dm = FaceDataModule()

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
