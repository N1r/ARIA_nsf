from aris.training.autoencoder import VoiceAutoEncoderCLI
from aris.training.callbacks import MyConfigCallback

if __name__ == "__main__":
    cli = VoiceAutoEncoderCLI(
        # VoiceAutoEncoder,
        # subclass_mode_model=True,
        trainer_defaults={
            "accelerator": "auto",
            "strategy": "auto",
            "devices": 1,
            "log_every_n_steps": 1,
        },
        save_config_callback=MyConfigCallback,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
    )
