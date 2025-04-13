
import os
import gc
from time import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Third party libraries
import torch
from dataset import generate_transforms, generate_dataloaders
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# User defined libraries
from models import se_resnext50_32x4d
from utils import init_hparams, init_logger, seed_reproducer, load_data
from loss_function import CrossEntropyLossOneHot
from lrs_scheduler import WarmRestart, warm_restart


class CoolSystem(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        seed_reproducer(self.hparams.seed)

        self.model = se_resnext50_32x4d()
        self.criterion = CrossEntropyLossOneHot()
        self.logger_kun = init_logger("kun_in", hparams.log_dir)

        # Хранилище для train и validation outputs
        self.train_step_outputs = []  # Инициализируем этот атрибут
        self.validation_step_outputs = []  # Уже инициализировано в предыдущем ответе

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.scheduler = WarmRestart(self.optimizer, T_max=10, T_mult=1, eta_min=1e-5)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        step_start_time = time()
        images, labels, data_load_time = batch

        scores = self(images)
        loss = self.criterion(scores, labels)
        data_load_time = torch.sum(data_load_time)
        batch_run_time = torch.Tensor([time() - step_start_time + data_load_time]).to(data_load_time.device)

        output = {
            "loss": loss,
            "data_load_time": data_load_time,
            "batch_run_time": batch_run_time,
        }
        self.train_step_outputs.append(output)  # Теперь этот атрибут существует
        return loss

    def on_train_epoch_end(self):
        outputs = self.train_step_outputs
        train_loss_mean = torch.stack([o["loss"] for o in outputs]).mean()
        self.data_load_times = torch.stack([o["data_load_time"] for o in outputs]).sum()
        self.batch_run_times = torch.stack([o["batch_run_time"] for o in outputs]).sum()

        if self.current_epoch < (self.trainer.max_epochs - 4):
            self.scheduler = warm_restart(self.scheduler, T_mult=2)

        self.log("train_loss", train_loss_mean, prog_bar=True)
        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        step_start_time = time()
        images, labels, data_load_time = batch
        data_load_time = torch.sum(data_load_time)
        scores = self(images)
        loss = self.criterion(scores, labels)
        batch_run_time = torch.Tensor([time() - step_start_time + data_load_time]).to(data_load_time.device)

        output = {
            "val_loss": loss,
            "scores": scores,
            "labels": labels,
            "data_load_time": data_load_time,
            "batch_run_time": batch_run_time,
        }
        self.validation_step_outputs.append(output)

    def on_validation_epoch_end(self):
        # Проверка на пустоту outputs
        if len(self.validation_step_outputs) == 0:
            print("Warning: No validation outputs collected.")
            return

        val_loss_mean = torch.stack([o["val_loss"] for o in self.validation_step_outputs]).mean()
        self.data_load_times = torch.stack([o["data_load_time"] for o in self.validation_step_outputs]).sum()
        self.batch_run_times = torch.stack([o["batch_run_time"] for o in self.validation_step_outputs]).sum()

        scores_all = torch.cat([o["scores"] for o in self.validation_step_outputs]).cpu()
        labels_all = torch.round(torch.cat([o["labels"] for o in self.validation_step_outputs]).cpu())
        val_roc_auc = roc_auc_score(labels_all, scores_all)

        self.logger_kun.info(
            f"{self.hparams.fold_i}-{self.current_epoch} | "
            f"lr : {self.scheduler.get_lr()[0]:.6f} | "
            f"val_loss : {val_loss_mean:.4f} | "
            f"val_roc_auc : {val_roc_auc:.4f} | "
            f"data_load_times : {self.data_load_times:.2f} | "
            f"batch_run_times : {self.batch_run_times:.2f}"
        )

        self.log("val_loss", val_loss_mean, prog_bar=True)
        self.log("val_roc_auc", val_roc_auc, prog_bar=True)

        self.validation_step_outputs.clear()





if __name__ == "__main__":
    # Make experiment reproducible
    seed_reproducer(2020)

    # Init Hyperparameters
    hparams = init_hparams()

    # init logger
    logger = init_logger("kun_out", log_dir=hparams.log_dir)

    # Load data
    data, test_data = load_data(logger)

    # Generate transforms
    transforms = generate_transforms(hparams.image_size)

    # Do cross validation
    valid_roc_auc_scores = []
    folds = KFold(n_splits=5, shuffle=True, random_state=hparams.seed)
    for fold_i, (train_index, val_index) in enumerate(folds.split(data)):
        hparams.fold_i = fold_i
        train_data = data.iloc[train_index, :].reset_index(drop=True)
        val_data = data.iloc[val_index, :].reset_index(drop=True)

        train_dataloader, val_dataloader = generate_dataloaders(hparams, train_data, val_data, transforms)

        # Define callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val_roc_auc",
            save_top_k=6,
            mode="max",
            dirpath=os.path.join(hparams.log_dir, f"fold={fold_i}"),
            filename="{epoch}-{val_loss:.4f}-{val_roc_auc:.4f}",
        )
        early_stop_callback = EarlyStopping(monitor="val_roc_auc", patience=10, mode="max", verbose=True)

        # Instance Model, Trainer and train model
        model = CoolSystem(hparams)
        trainer = pl.Trainer(
    accelerator="gpu" if hparams.gpus else "cpu",
    devices=hparams.gpus if hparams.gpus else 1,
    min_epochs=70,
    max_epochs=hparams.max_epochs,
    callbacks=[early_stop_callback, checkpoint_callback],
    enable_progress_bar=False,
    precision=hparams.precision,
    num_sanity_val_steps=0,
    gradient_clip_val=hparams.gradient_clip_val,
)
        trainer.fit(model, train_dataloader, val_dataloader)

        valid_roc_auc_scores.append(round(checkpoint_callback.best_model_score, 4))

        logger.info(valid_roc_auc_scores)

        del model
        gc.collect()
        torch.cuda.empty_cache()
