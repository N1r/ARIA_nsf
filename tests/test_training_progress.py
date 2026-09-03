from types import SimpleNamespace

from aris.training.progress import TrainingProgressPrinter


def _trainer(*, step: int, train_loss: float = 1.25, val_loss: float = 0.75):
    return SimpleNamespace(
        is_global_zero=True,
        global_step=step,
        max_steps=100,
        current_epoch=1,
        sanity_checking=False,
        callback_metrics={"train_loss": train_loss, "val_loss": val_loss},
    )


def test_training_progress_prints_first_interval_and_last_step(capsys):
    callback = TrainingProgressPrinter(every_n_steps=50)

    for step in (1, 2, 50, 100):
        callback.on_train_batch_end(_trainer(step=step), None, None, None, 0)

    assert capsys.readouterr().out.splitlines() == [
        "[train] step 1/100 | epoch 2 | loss 1.2500",
        "[train] step 50/100 | epoch 2 | loss 1.2500",
        "[train] step 100/100 | epoch 2 | loss 1.2500",
    ]


def test_validation_progress_prints_loss(capsys):
    callback = TrainingProgressPrinter()

    callback.on_validation_epoch_end(_trainer(step=50), None)

    assert capsys.readouterr().out.strip() == ("[valid] step 50/100 | epoch 2 | loss 0.7500")
