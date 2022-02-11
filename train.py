import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import os
import torch
from data_utils import *
from pre_models import get_model
from models import *


def main():
    home = os.getcwd()
    args = load_arguments(home)
    mai_args = load_mai_arguments(home, args)
    print("Loading all arguments successfully!")

    training, validation, train_dataloader, val_dataloader = load_dataset(args, mai_args)

    model = get_model(args, training)
    print(f"Number of parameters in network: {model.size()/1e3:.1f}k")

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-16, patience=5, verbose=True, mode="min")
    lr_logger = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpu,  # run on CPU, if on multiple GPUs, use accelerator="ddp"
        gradient_clip_val=0.1,
        #limit_train_batches=30,
        #limit_val_batches=10,
        val_check_interval=0.25, # 每训练单个epoch的 % 调用校验函数一次
        #fast_dev_run=True,  # comment in to quickly check for bugs
        check_val_every_n_epoch=10,
        callbacks=[lr_logger, early_stop_callback],
        logger=TensorBoardLogger("lightning_logs")
    )
    

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
    )

    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    predictions = model.predict(val_dataloader)
    print(f"Mean absolute error of model: {(actuals - predictions).abs().mean()}")


if __name__ == '__main__':
    main()