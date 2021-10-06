#!/home/menno/Documents/GANdalf/.venv/bin/python
from types import SimpleNamespace
from pytorch_lightning import Trainer

from network.architecture import CycleGAN, StarGAN, StarGAN_2disc
from data import FaceDataModule


def main():
    args = {'mask_type': 'mask', 'block_type': 'resb',
            'disc_type': 'default', 'gen_type': 'default',
            'dilation': 1, }
    args = SimpleNamespace(**args)
    model = CycleGAN.load_from_checkpoint(
        checkpoint_path="lightning_logs/version_38/checkpoints/CycleGAN-epoch01-lpips0.08.ckpt",
        strict=False,
        args=args,
    )
    trainer = Trainer(logger=False, gpus=1)
    dm = FaceDataModule()

    trainer.test(model, dm)


if __name__ == "__main__":
    main()
