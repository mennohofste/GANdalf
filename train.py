from pytorch_lightning import Trainer
from network.architecture import CycleGAN
from data import FaceDataModule


def main():
    model = CycleGAN()
    trainer = Trainer()
    dm = FaceDataModule()

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
