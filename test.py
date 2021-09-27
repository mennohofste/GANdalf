#!/home/menno/Documents/GANdalf/.venv/bin/python
from pytorch_lightning import Trainer

from network.architecture import CycleGAN
from data import FaceDataModule


def main():
    model = CycleGAN.load_from_checkpoint(
        checkpoint_path="lightning_logs/version_1/checkpoints/CycleGAN-epoch01-l10.05.ckpt",
        args=None,
    )
    trainer = Trainer(logger=False)
    dm = FaceDataModule()

    trainer.test(model, dm)


if __name__ == "__main__":
    main()
