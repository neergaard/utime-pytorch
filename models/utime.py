from argparse import ArgumentParser

# import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn
# import wandb
#from pytorch_lightning import EvalResult, TrainResult
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy
from sklearn import metrics
from tqdm import tqdm

import utils
# from utils.plotting import plot_segment


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels=5, out_channels=5, kernel_size=3, dilation=1, activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.padding = (self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1) // 2

        self.layers = nn.Sequential(
            nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_channels),
        )
        nn.init.xavier_uniform_(self.layers[1].weight)
        nn.init.zeros_(self.layers[1].bias)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, filters=[16, 32, 64, 128], in_channels=5, maxpool_kernels=[10, 8, 6, 4], kernel_size=5, dilation=2):
        super().__init__()
        self.filters = filters
        self.in_channels = in_channels
        self.maxpool_kernels = maxpool_kernels
        self.kernel_size = kernel_size
        self.dilation = dilation
        assert len(self.filters) == len(
            self.maxpool_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied maxpool kernels ({len(self.maxpool_kernels)})!"

        self.depth = len(self.filters)

        # fmt: off
        self.blocks = nn.ModuleList([nn.Sequential(
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                activation="relu",
            ),
            ConvBNReLU(
                in_channels=self.filters[k],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                activation="relu",
            ),
        ) for k in range(self.depth)])
        # fmt: on

        self.maxpools = nn.ModuleList([nn.MaxPool1d(self.maxpool_kernels[k]) for k in range(self.depth)])

        self.bottom = nn.Sequential(
            ConvBNReLU(
                in_channels=self.filters[-1],
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size,
            ),
            ConvBNReLU(
                in_channels=self.filters[-1] * 2,
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size
            ),
        )

    def forward(self, x):
        shortcuts = []
        for layer, maxpool in zip(self.blocks, self.maxpools):
            z = layer(x)
            shortcuts.append(z)
            x = maxpool(z)

        # Bottom part
        encoded = self.bottom(x)

        return encoded, shortcuts


class Decoder(nn.Module):
    def __init__(self, filters=[128, 64, 32, 16], upsample_kernels=[4, 6, 8, 10], in_channels=256, out_channels=5, kernel_size=5):
        super().__init__()
        self.filters = filters
        self.upsample_kernels = upsample_kernels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        assert len(self.filters) == len(
            self.upsample_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied upsample kernels ({len(self.upsample_kernels)})!"
        self.depth = len(self.filters)

        # fmt: off
        self.upsamples = nn.ModuleList([nn.Sequential(
            nn.Upsample(scale_factor=self.upsample_kernels[k]),
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                activation='relu',
            )
        ) for k in range(self.depth)])

        self.blocks = nn.ModuleList([nn.Sequential(
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
            ),
            ConvBNReLU(
                in_channels=self.filters[k],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
            ),
        ) for k in range(self.depth)])
        # fmt: off

    def forward(self, z, shortcuts):

        for upsample, block, shortcut in zip(self.upsamples, self.blocks, shortcuts[::-1]):
            z = upsample(z)
            z = torch.cat([shortcut, z], dim=1)
            z = block(z)

        return z


class SegmentClassifier(nn.Module):
    def __init__(self, sampling_frequency=128, num_classes=5, epoch_length=30):
        super().__init__()
        # self.sampling_frequency = sampling_frequency
        self.num_classes = num_classes
        # self.epoch_length = epoch_length
        self.layers = nn.Sequential(
            # nn.AvgPool2d(kernel_size=(1, self.epoch_length * self.sampling_frequency)),
            # nn.Flatten(start_dim=2),
            # nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(in_channels=self.num_classes, out_channels=self.num_classes, kernel_size=1),
            nn.Softmax(dim=1),
        )
        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.zeros_(self.layers[0].bias)

    def forward(self, x):
        # batch_size, num_classes, n_samples = x.shape
        # z = x.reshape((batch_size, num_classes, -1, self.epoch_length * self.sampling_frequency))
        return self.layers(x)


class UTimeModel(LightningModule):
    # def __init__(
    #     self, filters=[16, 32, 64, 128], in_channels=5, maxpool_kernels=[10, 8, 6, 4], kernel_size=5,
    #     dilation=2, sampling_frequency=128, num_classes=5, epoch_length=30, lr=1e-4, batch_size=12,
    #     n_workers=0, eval_ratio=0.1, data_dir=None, n_jobs=-1, n_records=-1, scaling=None, **kwargs
    # ):
    def __init__(
        self,
        filters=None,
        in_channels=None,
        maxpool_kernels=None,
        kernel_size=None,
        dilation=None,
        num_classes=None,
        sampling_frequency=None,
        epoch_length=None,
        data_dir=None,
        n_jobs=None,
        n_records=None,
        scaling=None,
        lr=None,
        n_segments=10,
        *args,
        **kwargs
    ):
    # def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # self.save_hyperparameters(hparams)
        # self.save_hyperparameters({k: v for k, v in hparams.items() if not callable(v)})
        self.encoder = Encoder(
            filters=self.hparams.filters,
            in_channels=self.hparams.in_channels,
            maxpool_kernels=self.hparams.maxpool_kernels,
            kernel_size=self.hparams.kernel_size,
            dilation=self.hparams.dilation,
        )
        self.decoder = Decoder(
            filters=self.hparams.filters[::-1],
            upsample_kernels=self.hparams.maxpool_kernels[::-1],
            in_channels=self.hparams.filters[-1] * 2,
            kernel_size=self.hparams.kernel_size,
        )
        self.dense = nn.Sequential(
            nn.Conv1d(in_channels=self.hparams.filters[0], out_channels=self.hparams.num_classes, kernel_size=1, bias=True),
            nn.Tanh()
        )
        nn.init.xavier_uniform_(self.dense[0].weight)
        nn.init.zeros_(self.dense[0].bias)
        self.segment_classifier = SegmentClassifier(
            sampling_frequency=self.hparams.sampling_frequency,
            num_classes=self.hparams.num_classes,
            epoch_length=self.hparams.epoch_length
        )
        self.loss = utils.DiceLoss(self.hparams.num_classes)
        # self.loss = nn.CrossEntropyLoss()
        # self.train_acc = Accuracy()
        # self.eval_acc = Accuracy()
#        self.metric = Accuracy(num_classes=self.hparams.num_classes, reduce_op='mean')

        # Create Dataset params
        self.dataset_params = dict(
            data_dir=self.hparams.data_dir,
            n_jobs=self.hparams.n_jobs,
            n_records=self.hparams.n_records,
            scaling=self.hparams.scaling,
        )

        # # Create DataLoader params
        # self.dataloader_params = dict(
        #     batch_size=self.hparams.batch_size,
        #     num_workers=self.hparams.n_workers,
        #     pin_memory=True,
        # )

        # Create Optimizer params
        self.optimizer_params = dict(lr=self.hparams.lr)
        # self.example_input_array = torch.zeros(1, self.hparams.in_channels, 35 * 30 * 100)

    def forward(self, x):

        # Run through encoder
        z, shortcuts = self.encoder(x)

        # Run through decoder
        z = self.decoder(z, shortcuts)

        # Run dense modeling
        z = self.dense(z)

        return z

    def classify_segments(self, x, resolution=30):

        # Run through encoder + decoder
        z = self(x)

        # Classify decoded samples
        resolution_samples = self.hparams.sampling_frequency * resolution
        z = z.unfold(-1, resolution_samples, resolution_samples) \
             .mean(dim=-1)
        y = self.segment_classifier(z)

        return y

    def training_step(self, batch, batch_idx):
        # if batch_idx == 100:
        #     print('hej')
        x, t, r, seq, stable_sleep = batch

        # Classify segments
        y = self.classify_segments(x)

        # loss = self.loss(y, t[:, :, ::self.hparams.epoch_length])
        loss = self.compute_loss(y, t, stable_sleep)
        # self.train_acc(y.argmax(dim=1)[ss], t_.argmax(dim=1)[ss])
        # accuracy = self.metric(y.argmax(dim=1), t[:, :, ::self.hparams.epoch_length].argmax(dim=1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {
            'loss': loss,
            'predicted': y,
            'true': t,
            'record': r,
            'sequence_nr': seq,
            'stable_sleep': stable_sleep
        }
#        result = TrainResult(minimize=loss)
#        result.log('train_loss', loss, prog_bar=True, sync_dist=True)
#        result.log('train_acc', accuracy, prog_bar=True, sync_dist=True)
#        return result

    def validation_step(self, batch, batch_idx):
        x, t, r, seq, stable_sleep = batch

        # Classify segments
        y = self.classify_segments(x)

        # loss = self.loss(y, t[:, :, ::self.hparams.epoch_length])
        loss = self.compute_loss(y, t, stable_sleep)
        # self.eval_acc(y.argmax(dim=1)[ss], t_.argmax(dim=1)[ss])
        # accuracy = self.metric(y.argmax(dim=1), t[:, :, ::self.hparams.epoch_length].argmax(dim=1))
        self.log('eval_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        # self.log('eval_acc', self.eval_acc, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return {
            'loss': loss,
            'predicted': y,
            'true': t,
            'record': r,
            'sequence_nr': seq,
            'stable_sleep': stable_sleep,
        }
        # # Generate an image
        # if batch_idx == 0:
        #     fig = plot_segment(x, t, z)
        #     self.logger.experiment[1].log({'Hypnodensity': wandb.Image(fig)})
        #     plt.close(fig)

#        result = EvalResult(checkpoint_on=loss)
#        result.log('eval_loss', loss, prog_bar=True, sync_dist=True)
#        result.log('eval_acc', accuracy, prog_bar=True, sync_dist=True)
#
#        return result

    def test_step(self, batch, batch_index):

        X, t, current_record, current_sequence, stable_sleep = batch
        y = self.classify_segments(X)
        y_1s = self.classify_segments(X, resolution=1)
        # result = ptl.EvalResult()
        # result.predicted = y_hat.softmax(dim=1)
        # result.true = y
        # result.record = current_record
        # result.sequence_nr = current_sequence
        # result.stable_sleep = stable_sleep
        # return result
        return {
            "predicted": y,
            "true": t,
            "record": current_record,
            "sequence_nr": current_sequence,
            "stable_sleep": stable_sleep,
            'logits': y_1s
        }

    def training_epoch_end(self, outputs):

        true = torch.cat([out['true'] for out in outputs], dim=0).permute([0, 2, 1])
        predicted = torch.cat([out['predicted'] for out in outputs], dim=0).permute([0, 2, 1])
        stable_sleep = torch.cat([out['stable_sleep'].to(torch.int64) for out in outputs], dim=0)
        sequence_nrs = torch.cat([out['sequence_nr'] for out in outputs], dim=0)

        if self.use_ddp:
            out_true = [torch.zeros_like(true) for _ in range(torch.distributed.get_world_size())]
            out_predicted = [torch.zeros_like(predicted) for _ in range(torch.distributed.get_world_size())]
            out_stable_sleep = [torch.zeros_like(stable_sleep) for _ in range(torch.distributed.get_world_size())]
            out_seq_nrs = [torch.zeros_like(sequence_nrs) for _ in range(dist.get_world_size())]
            dist.barrier()
            dist.all_gather(out_true, true)
            dist.all_gather(out_predicted, predicted)
            dist.all_gather(out_stable_sleep, stable_sleep)
            dist.all_gather(out_seq_nrs, sequence_nrs)
            if dist.get_rank() == 0:
                t = torch.stack(out_true).transpose(0, 1).reshape(-1, int(300/self.hparams.eval_frequency_sec), self.hparams.n_classes).cpu().numpy()
                p = torch.stack(out_predicted).transpose(0, 1).reshape(-1, int(300/self.hparams.eval_frequency_sec), self.hparams.n_classes).cpu().numpy()
                s = torch.stack(out_stable_sleep).transpose(0, 1).reshape(-1, int(300/self.hparams.eval_frequency_sec)).to(torch.bool).cpu().numpy()
                u = t.sum(axis=-1) == 1

                acc = metrics.accuracy_score(t[s & u].argmax(-1), p[s & u].argmax(-1))
                cohen = metrics.cohen_kappa_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4])
                f1_macro = metrics.f1_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average='macro')

                self.log_dict({
                    'train_acc': acc,
                    'train_cohen': cohen,
                    'train_f1_macro': f1_macro,
                }, on_step=False, prog_bar=True, logger=True, on_epoch=True)
        elif self.on_gpu:
            t = true.cpu().numpy()
            p = predicted.cpu().detach().numpy()
            s = stable_sleep.to(torch.bool).cpu().numpy()
            u = t.sum(axis=-1) == 1

            acc = metrics.accuracy_score(t[s & u].argmax(-1), p[s & u].argmax(-1))
            cohen = metrics.cohen_kappa_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4])
            f1_macro = metrics.f1_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average='macro')

            self.log_dict({
                    'train_acc': acc,
                    'train_cohen': cohen,
                    'train_f1_macro': f1_macro,
            }, on_step=False, prog_bar=True, logger=True, on_epoch=True)
        else:
            t = true.numpy()
            p = predicted.numpy()
            s = stable_sleep.to(torch.bool).numpy()
            u = t.sum(axis=-1) == 1

            acc = metrics.accuracy_score(t[s & u].argmax(-1), p[s & u].argmax(-1))
            cohen = metrics.cohen_kappa_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4])
            f1_macro = metrics.f1_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average='macro')

            self.log_dict({
                    'train_acc': acc,
                    'train_cohen': cohen,
                    'train_f1_macro': f1_macro,
            }, on_step=False, prog_bar=True, logger=True, on_epoch=True)

    def validation_epoch_end(self, outputs):

        true = torch.cat([out['true'] for out in outputs], dim=0).permute([0, 2, 1])
        predicted = torch.cat([out['predicted'] for out in outputs], dim=0).permute([0, 2, 1])
        stable_sleep = torch.cat([out['stable_sleep'].to(torch.int64) for out in outputs], dim=0)
        sequence_nrs = torch.cat([out['sequence_nr'] for out in outputs], dim=0)

        if self.use_ddp:
            out_true = [torch.zeros_like(true) for _ in range(torch.distributed.get_world_size())]
            out_predicted = [torch.zeros_like(predicted) for _ in range(torch.distributed.get_world_size())]
            out_stable_sleep = [torch.zeros_like(stable_sleep) for _ in range(torch.distributed.get_world_size())]
            out_seq_nrs = [torch.zeros_like(sequence_nrs) for _ in range(dist.get_world_size())]
            dist.barrier()
            dist.all_gather(out_true, true)
            dist.all_gather(out_predicted, predicted)
            dist.all_gather(out_stable_sleep, stable_sleep)
            dist.all_gather(out_seq_nrs, sequence_nrs)
            if dist.get_rank() == 0:
                t = torch.stack(out_true).transpose(0, 1).reshape(-1, int(300/self.hparams.eval_frequency_sec), self.hparams.n_classes).cpu().numpy()
                p = torch.stack(out_predicted).transpose(0, 1).reshape(-1, int(300/self.hparams.eval_frequency_sec), self.hparams.n_classes).cpu().numpy()
                s = torch.stack(out_stable_sleep).transpose(0, 1).reshape(-1, int(300/self.hparams.eval_frequency_sec)).to(torch.bool).cpu().numpy()
                u = t.sum(axis=-1) == 1

                acc = metrics.accuracy_score(t[s & u].argmax(-1), p[s & u].argmax(-1))
                cohen = metrics.cohen_kappa_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4])
                f1_macro = metrics.f1_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average='macro')

                self.log_dict({
                    'eval_acc': acc,
                    'eval_cohen': cohen,
                    'eval_f1_macro': f1_macro,
                }, on_step=False, prog_bar=True, logger=True, on_epoch=True)
        elif self.on_gpu:
            t = true.cpu().numpy()
            p = predicted.cpu().numpy()
            s = stable_sleep.to(torch.bool).cpu().numpy()
            u = t.sum(axis=-1) == 1

            acc = metrics.accuracy_score(t[s & u].argmax(-1), p[s & u].argmax(-1))
            cohen = metrics.cohen_kappa_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4])
            f1_macro = metrics.f1_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average='macro')

            self.log_dict({
                    'eval_acc': acc,
                    'eval_cohen': cohen,
                    'eval_f1_macro': f1_macro,
            }, on_step=False, prog_bar=True, logger=True, on_epoch=True)
        else:
            t = true.numpy()
            p = predicted.numpy()
            s = stable_sleep.to(torch.bool).numpy()
            u = t.sum(axis=-1) == 1

            acc = metrics.accuracy_score(t[s & u].argmax(-1), p[s & u].argmax(-1))
            cohen = metrics.cohen_kappa_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4])
            f1_macro = metrics.f1_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average='macro')

            self.log_dict({
                    'eval_acc': acc,
                    'eval_cohen': cohen,
                    'eval_f1_macro': f1_macro,
            }, on_step=False, prog_bar=True, logger=True, on_epoch=True)

    def test_epoch_end(self, outputs):
        """This method collects the results and sorts the predictions according to record and sequence nr."""

        try:
            all_records = sorted(self.trainer.datamodule.test.records)
        except AttributeError: # Catch exception if we've supplied dataloaders instead of DataModule
            all_records = sorted(self.trainer.test_dataloaders[0].dataset.records)

        true = torch.cat([out['true'] for out in outputs], dim=0).permute([0, 2, 1])
        predicted = torch.cat([out['predicted'] for out in outputs], dim=0).permute([0, 2, 1])
        stable_sleep = torch.cat([out['stable_sleep'].to(torch.int64) for out in outputs], dim=0)
        sequence_nrs = torch.cat([out['sequence_nr'] for out in outputs], dim=0)
        logits = torch.cat([out['logits'] for out in outputs], dim=0).permute([0, 2, 1])

        if self.use_ddp:
            record2int = {r: idx for idx, r in enumerate(all_records)}
            int2record = {idx: r for idx, r in enumerate(all_records)}
            records = torch.cat([torch.Tensor([record2int[r]]).type_as(stable_sleep) for out in outputs for r in out['record']], dim=0)
            out_true = [torch.zeros_like(true) for _ in range(torch.distributed.get_world_size())]
            out_predicted = [torch.zeros_like(predicted) for _ in range(torch.distributed.get_world_size())]
            out_stable_sleep = [torch.zeros_like(stable_sleep) for _ in range(torch.distributed.get_world_size())]
            out_seq_nrs = [torch.zeros_like(sequence_nrs) for _ in range(dist.get_world_size())]
            out_records = [torch.zeros_like(records) for _ in range(dist.get_world_size())]
            out_logits = [torch.zeros_like(logits) for _ in range(dist.get_world_size())]

            dist.barrier()
            dist.all_gather(out_true, true)
            dist.all_gather(out_predicted, predicted)
            dist.all_gather(out_stable_sleep, stable_sleep)
            dist.all_gather(out_seq_nrs, sequence_nrs)
            dist.all_gather(out_records, records)
            dist.all_gather(out_logits, logits)

            if dist.get_rank() == 0:
                true = torch.cat(out_true)
                predicted = torch.cat(out_predicted)
                stable_sleep = torch.cat(out_stable_sleep)
                sequence_nrs = torch.cat(out_seq_nrs)
                records = [int2record[r.item()] for r in torch.cat(out_records)]
                logits = torch.cat(out_logits)
                # t = torch.stack(out_true).transpose(0, 1).reshape(-1, int(300/self.hparams.eval_frequency_sec), self.hparams.n_classes).cpu().numpy()
                # p = torch.stack(out_predicted).transpose(0, 1).reshape(-1, int(300/self.hparams.eval_frequency_sec), self.hparams.n_classes).cpu().numpy()
                # s = torch.stack(out_stable_sleep).transpose(0, 1).reshape(-1, int(300/self.hparams.eval_frequency_sec)).to(torch.bool).cpu().numpy()
                # u = t.sum(axis=-1) == 1

            else:
                return None
        else:
            records = [r for out in outputs for r in out['record']]

        # elif self.on_gpu:
        #     t = true.cpu().numpy()[:, ::self.hparams.epoch_length, :]
        #     p = predicted.cpu().numpy()
        #     s = stable_sleep.to(torch.bool).cpu().numpy()[:, ::self.hparams.epoch_length]
        #     u = t.sum(axis=-1) == 1
        # else:
        #     t = true.numpy()[:, ::self.hparams.epoch_length, :]
        #     p = predicted.numpy()
        #     s = stable_sleep.to(torch.bool).numpy()[:, ::self.hparams.epoch_length, :]
        #     u = t.sum(axis=-1) == 1

        results = {
            r: {
                "true": [],
                "true_label": [],
                "predicted": [],
                "predicted_label": [],
                "stable_sleep": [],
                "logits": [],
                "seq_nr": [],
                # 'acc': None,
                # 'f1': None,
                # 'recall': None,
                # 'precision': None,
            } for r in all_records
        }

        for r in tqdm(all_records, desc='Sorting predictions...'):
            record_idx = [idx for idx, rec in enumerate(records) if r == rec]
            current_t = torch.cat([t for idx, t in enumerate(true) if idx in record_idx], dim=0).cpu().numpy()
            current_p = torch.cat([p for idx, p in enumerate(predicted) if idx in record_idx], dim=0).cpu().numpy()
            current_ss = torch.cat([ss.to(torch.bool) for idx, ss in enumerate(stable_sleep) if idx in record_idx], dim=0).cpu().numpy()
            current_l = torch.cat([l for idx, l in enumerate(logits) if idx in record_idx], dim=0).cpu().numpy()
            current_seq = torch.stack([seq for idx, seq in enumerate(sequence_nrs) if idx in record_idx]).cpu().numpy()

            results[r]['true'] = current_t.reshape(-1, self.hparams.n_segments, self.hparams.num_classes)[current_seq.argsort()].reshape(-1, self.hparams.num_classes)
            results[r]['predicted'] = current_p.reshape(-1, self.hparams.n_segments, self.hparams.num_classes)[current_seq.argsort()].reshape(-1, self.hparams.num_classes)
            results[r]['stable_sleep'] = current_ss.reshape(-1, self.hparams.n_segments)[current_seq.argsort()].reshape(-1)
            results[r]['logits'] = current_l.reshape(-1, self.hparams.epoch_length * self.hparams.n_segments, self.hparams.num_classes)[current_seq.argsort()].reshape(-1, self.hparams.num_classes).shape
            results[r]['sequence'] = current_seq[current_seq.argsort()]

        return results

    def compute_loss(self, y_pred, y_true, stable_sleep):
        # stable_sleep = stable_sleep[:, ::self.hparams.epoch_length]
        # y_true = y_true[:, :, ::self.hparams.epoch_length]

        if y_pred.shape[-1] != self.hparams.num_classes:
            y_pred = y_pred.permute(dims=[0, 2, 1])
        if y_true.shape[-1] != self.hparams.num_classes:
            y_true = y_true.permute(dims=[0, 2, 1])
        # return self.loss(y_pred, y_true.argmax(dim=-1))

        # return
        return self.loss(y_pred, y_true, stable_sleep)

    def configure_optimizers(self):
        return torch.optim.Adam(
            # [
                # {'params': list(self.encoder.parameters())},
                # {'params': list(self.decoder.parameters())},
                # # {'params': [p[1] for p in self.named_parameters() if 'bias' not in p[0] and 'batch_norm' not in p[0]]},
                # {'params': list(self.segment_classifier.parameters())[0], 'weight_decay': 1e-5},
                # {'params': list(self.segment_classifier.parameters())[1]},
            # ],
            self.parameters(), **self.optimizer_params
        )

    # def on_after_backward(self):
    #     print('Hej')

    # def train_dataloader(self):
    #     """Return training dataloader."""
    #     return DataLoader(self.train_data, shuffle=True, **self.dataloader_params)

    # def val_dataloader(self):
    #     """Return validation dataloader."""
    #     return DataLoader(self.eval_data, shuffle=False, **self.dataloader_params)

    # def setup(self, stage):
    #     if stage == 'fit':
    #         self.dataset = SscWscPsgDataset(**self.dataset_params)
    #         self.train_data, self.eval_data = self.dataset.split_data(self.hparams.eval_ratio)

    @staticmethod
    def add_model_specific_args(parent_parser):

        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=True)
        architecture_group = parser.add_argument_group('architecture')
        architecture_group.add_argument('--filters', default=[16, 32, 64, 128], nargs='+', type=int)
        architecture_group.add_argument('--in_channels', default=5, type=int)
        architecture_group.add_argument('--maxpool_kernels', default=[10, 8, 6, 4], nargs='+', type=int)
        architecture_group.add_argument('--kernel_size', default=5, type=int)
        architecture_group.add_argument('--dilation', default=2, type=int)
        architecture_group.add_argument('--sampling_frequency', default=128, type=int)
        architecture_group.add_argument('--num_classes', default=5, type=int)
        architecture_group.add_argument('--epoch_length', default=30, type=int)
        architecture_group.add_argument('--n_segments', default=10, type=int)

        # OPTIMIZER specific
        optimizer_group = parser.add_argument_group('optimizer')
        # optimizer_group.add_argument('--optimizer', default='sgd', type=str)
        optimizer_group.add_argument('--lr', default=5e-6, type=float)
        # optimizer_group.add_argument('--momentum', default=0.9, type=float)
        # optimizer_group.add_argument('--weight_decay', default=0, type=float)

        # LEARNING RATE SCHEDULER specific
        # lr_scheduler_group = parser.add_argument_group('lr_scheduler')
        # lr_scheduler_group.add_argument('--lr_scheduler', default=None, type=str)
        # lr_scheduler_group.add_argument('--base_lr', default=0.05, type=float)
        # lr_scheduler_group.add_argument('--lr_reduce_factor', default=0.1, type=float)
        # lr_scheduler_group.add_argument('--lr_reduce_patience', default=5, type=int)
        # lr_scheduler_group.add_argument('--max_lr', default=0.15, type=float)
        # lr_scheduler_group.add_argument('--step_size_up', default=0.05, type=int)

        # DATASET specific
        # dataset_group = parser.add_argument_group('dataset')
        # dataset_group.add_argument('--data_dir', default='data/train/raw/individual_encodings', type=str)
        # dataset_group.add_argument('--eval_ratio', default=0.1, type=float)
        # dataset_group.add_argument('--n_jobs', default=-1, type=int)
        # dataset_group.add_argument('--n_records', default=-1, type=int)
        # dataset_group.add_argument('--scaling', default=None, type=str)


        # DATALOADER specific
        # dataloader_group = parser.add_argument_group('dataloader')
        # dataloader_group.add_argument('--batch_size', default=12, type=int)
        # dataloader_group.add_argument('--n_workers', default=0, type=int)

        return parser


if __name__ == "__main__":
    from datasets import SscWscDataModule
    from pytorch_lightning.core.memory import ModelSummary

    parser = ArgumentParser(add_help=False)
    parser = SscWscDataModule.add_dataset_specific_args(parser)
    parser = UTimeModel.add_model_specific_args(parser)
    args = parser.parse_args()
    # parser.add_argument('--filters', default=[16, 32, 64, 128], nargs='+', type=int)
    # args = parser.parse_args()
    # print('Filters:', args.filters)
    args.in_channels = 4
    in_channels = args.in_channels
    x_shape = (1, in_channels, 10 * 30 * 128)
    x = torch.rand(x_shape)

    # # Test ConvBNReLU block
    # z = ConvBNReLU()(x)
    # print()
    # print(ConvBNReLU())
    # print(x.shape)
    # print(z.shape)

    # # test Encoder class
    # encoder = Encoder()
    # print(encoder)
    # print("x.shape:", x.shape)
    # z, shortcuts = encoder(x)
    # print("z.shape:", z.shape)
    # print("Shortcuts shape:", [shortcut.shape for shortcut in shortcuts])

    # # Test Decoder class
    # z_shape = (32, 256, 54)
    # z = torch.rand(z_shape)
    # decoder = Decoder()
    # print(decoder)
    # x_hat = decoder(z, None)
    # print("x_hat.shape:", x_hat.shape)

    # Test UTimeModel Class
    # utime = UTimeModel(in_channels=in_channels)
    utime = UTimeModel(vars(args))
    utime.example_input_array = torch.zeros(x_shape)
    utime.configure_optimizers()
    model_summary = ModelSummary(utime, "full")
    print(model_summary)
    print(utime)
    print(x.shape)
    # z = utime(x)
    z = utime.classify_segments(x)
    print(z.shape)
    print("x.shape:", x.shape)
    print("z.shape:", z.shape)
    print(z.sum(dim=1))
