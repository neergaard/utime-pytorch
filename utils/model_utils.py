from argparse import Namespace

import torch

import models


def get_model(args):

    if args.resume_from_checkpoint:
        # resume_from_checkpoint = args.resume_from_checkpoint
        # args = Namespace(
        #     **(torch.load(args.resume_from_checkpoint, map_location=torch.device("cpu"))["hyper_parameters"])
        # )
        # args.resume_from_checkpoint = resume_from_checkpoint
        model = models.available_models[args.model_type].load_from_checkpoint(args.resume_from_checkpoint)
    else:
        model = models.available_models[args.model_type](**vars(args))

    return model
