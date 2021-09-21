#!/home/menno/Documents/GANdalf/.venv/bin/python
from pytorch_lightning import Trainer

from network.architecture import CycleGAN
from network.architecture import StarGAN
from data import FaceDataModule


def main():
    model = CycleGAN.load_from_checkpoint(
        checkpoint_path="checkpoints/CycleGAN-epoch03-l10.18.ckpt"
    )
    trainer = Trainer()
    dm = FaceDataModule()

    trainer.test(model, dm)


if __name__ == "__main__":
    main()
