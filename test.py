#!/home/menno/Documents/GANdalf/.venv/bin/python
from types import SimpleNamespace
from pytorch_lightning import Trainer

from network.architecture import CycleGAN, StarGAN, StarGAN_2disc
from data import FaceDataModule


def main():
    args = {'mask_type': 'no', 'disc_type': 'PatchGAN', 'gen_type': 'default'}
    args = SimpleNamespace(**args)
    model = CycleGAN.load_from_checkpoint(
        checkpoint_path="lightning_logs/version_24/checkpoints/CycleGAN-epoch01-lpips0.25.ckpt",
        args=args,
    )
    trainer = Trainer(logger=False, gpus=1)
    dm = FaceDataModule()

    trainer.test(model, dm)


if __name__ == "__main__":
    main()
