import os

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks


def get_loggers_callbacks(args, model=None):

    try:
        # Setup logger(s) params
        csv_logger_params = dict(
            save_dir="./experiments",
            name=os.path.join(*args.save_dir.split("/")[1:-1]),
            version=args.save_dir.split("/")[-1],
        )
        wandb_logger_params = dict(
            log_model=False,
            name=os.path.join(*args.save_dir.split("/")[1:]),
            offline=args.debug,
            project="utime",
            save_dir=args.save_dir,
        )
        loggers = [
            pl_loggers.CSVLogger(**csv_logger_params),
            pl_loggers.WandbLogger(**wandb_logger_params),
        ]
        if model:
            loggers[-1].watch(model)

        # Setup callback(s) params
        checkpoint_monitor_params = dict(
            filepath=os.path.join(args.save_dir, "{epoch:03d}-{eval_loss:.2f}"),
            monitor=args.checkpoint_monitor,
            save_last=True,
            save_top_k=1,
        )
        earlystopping_parameters = dict(monitor=args.earlystopping_monitor, patience=args.earlystopping_patience,)
        callbacks = [
            pl_callbacks.ModelCheckpoint(**checkpoint_monitor_params),
            pl_callbacks.EarlyStopping(**earlystopping_parameters),
            pl_callbacks.LearningRateMonitor(),
        ]

        return loggers, callbacks
    except AttributeError:
        return None, None
