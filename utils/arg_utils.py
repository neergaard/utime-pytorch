import argparse
import os
import pprint
from datetime import datetime
from glob import glob

import torch
import pytorch_lightning as pl

import datasets
import models


def get_args():

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lr_finder", action="store_true")
    parser.add_argument("--seed", default=1337, type=int)
    parser.add_argument("--dataset_type", type=str, default="ssc-wsc")
    parser.add_argument("--checkpoint_monitor", default="eval_loss", type=str)
    parser.add_argument("--earlystopping_monitor", default="eval_loss", type=str)
    parser.add_argument("--earlystopping_patience", default=100, type=int)

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # Check the supplied model type
    parser.add_argument("--model_type", type=str, default="utime")
    temp_args, _ = parser.parse_known_args()

    # Optionally resume from checkpoint
    if temp_args.resume_from_checkpoint and os.path.isdir(temp_args.resume_from_checkpoint):
        temp_args.resume_from_checkpoint = glob(os.path.join(temp_args.resume_from_checkpoint, "epoch*.ckpt"))[0]
    if temp_args.resume_from_checkpoint:
        hparams = torch.load(temp_args.resume_from_checkpoint, map_location=torch.device("cpu"))["hyper_parameters"]
        temp_args.model_type = hparams["model_type"]

    # add args from dataset
    parser = datasets.available_datasets["ssc-wsc"].add_dataset_specific_args(parser)

    # add args from model
    parser = models.available_models[temp_args.model_type].add_model_specific_args(parser)

    # parse params
    args = parser.parse_args()

    # update args from hparams
    if args.resume_from_checkpoint:
        args.model_type = hparams["model_type"]

    # Create a save directory
    if not args.resume_from_checkpoint:
        try:
            args.save_dir = os.path.join(
                "experiments", args.model_type, args.model_name, args.name, datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
        except AttributeError:
            args.save_dir = os.path.join("experiments", "utime", datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Get the best model from the directory by default
    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        args.resume_from_checkpoint = glob(os.path.join(args.resume_from_checkpoint, "epoch*.ckpt"))[0]

    # If you wish to view applied settings, uncomment these two lines.
    if args.debug:
        pprint.pprint(vars(args))

    return args
