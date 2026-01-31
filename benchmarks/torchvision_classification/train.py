# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
import time
import warnings

import helpers
import presets
import torch
import torch.utils.data
import torchvision
import utils
from torch import nn


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for image, target in metric_logger.log_every(data_loader, args.print_freq, header):
        if args.data_loading_only:
            continue

        start_time = time.time()
        image, target = image.to(device), target.to(device)

        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, args, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    metric_logger.add_meter("acc1", utils.SmoothedValue())
    metric_logger.add_meter("acc5", utils.SmoothedValue())

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            if args.data_loading_only:
                continue
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader, "dataset")
        and hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg


def create_data_loaders(args):
    print(f"file-system = {args.fs}")

    if args.fs == "fsx":
        dataset_dir = "/datasets01"
    elif args.fs == "fsx_isolated":
        dataset_dir = "/fsx_isolated"
    elif args.fs == "ontap":
        dataset_dir = "/datasets01_ontap"
    elif args.fs == "ontap_isolated":
        dataset_dir = "/ontap_isolated"
    else:
        raise ValueError(f"bad args.fs, got {args.fs}")

    dataset_dir += "/imagenet_full_size/061417/"
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )

    if args.no_transforms:
        train_preset = val_preset = helpers.no_transforms
    else:
        train_preset = presets.ClassificationPresetTrain(crop_size=train_crop_size)
        val_preset = presets.ClassificationPresetEval(crop_size=val_crop_size, resize_size=val_resize_size)

    if args.ds_type == "iterable":
        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_preset)
        train_dataset = helpers.MapStyleToIterable(train_dataset, shuffle=True)

        val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_preset)
        val_dataset = helpers.MapStyleToIterable(val_dataset, shuffle=False)

        train_sampler = val_sampler = None
        train_shuffle = None  # but actually True

    elif args.ds_type == "mapstyle":
        builder = helpers.PreLoadedMapStyle if args.preload_ds else torchvision.datasets.ImageFolder
        train_dataset = builder(train_dir, transform=train_preset)
        val_dataset = builder(val_dir, transform=val_preset)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        train_shuffle = None  # but actually True

    else:
        raise ValueError(f"Invalid value for args.ds_type ({args.ds_type})")

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_data_loader, val_data_loader, train_sampler


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print("\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items())))

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_data_loader, val_data_loader, train_sampler = create_data_loaders(args)

    num_classes = 1000  # I'm lazy. TODO change this

    print("Creating model")
    model = torchvision.models.__dict__[args.model](weights=args.weights, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        evaluate(model, criterion, val_data_loader, device=device, args=args)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, train_data_loader, device, epoch, args)
        lr_scheduler.step()
        evaluate(model, criterion, val_data_loader, device=device, args=args)

        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

        if epoch == 0:
            first_epoch_time = time.time() - start_time

    total_time = time.time() - start_time
    print(f"Training time: {datetime.timedelta(seconds=int(total_time))}")
    print(f"Training time (w/o 1st epoch): {datetime.timedelta(seconds=int(total_time - first_epoch_time))}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--fs", default="fsx", type=str)
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device (Use cuda or cpu Default: cuda)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=32,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=12,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 16)",
    )
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument(
        "--lr-step-size",
        default=30,
        type=int,
        help="decrease lr every step-size epochs",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")

    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")

    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )

    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only.",
    )
    parser.add_argument(
        "--val-resize-size",
        default=256,
        type=int,
        help="the resize size used for validation (default: 256)",
    )
    parser.add_argument(
        "--val-crop-size",
        default=224,
        type=int,
        help="the central crop size used for validation (default: 224)",
    )
    parser.add_argument(
        "--train-crop-size",
        default=224,
        type=int,
        help="the random crop size used for training (default: 224)",
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    parser.add_argument(
        "--ds-type",
        default="mapstyle",
        type=str,
        help="'iterable' or 'mapstyle' (for regular indexable datasets)",
    )

    parser.add_argument(
        "--preload-ds",
        action="store_true",
        help="whether to use a fake dataset where all images are pre-loaded in RAM and already transformed. "
        "Mostly useful to benchmark how fast a model training would be without data-loading bottlenecks."
        "Acc results are irrevant because we don't cache the entire dataset, only a very small fraction of it.",
    )
    parser.add_argument(
        "--data-loading-only",
        action="store_true",
        help="When on, we bypass the model's forward and backward passes. So mostly only the dataloading happens",
    )
    parser.add_argument(
        "--no-transforms",
        action="store_true",
        help="Whether to apply transforms to the images. No transforms means we "
        "load and decode PIL images as usual, but we don't transform them. Instead we discard them "
        "and the dataset will produce random tensors instead. We "
        "need to create random tensors because without transforms, the images would still be PIL images "
        "and they wouldn't be of the required size."
        "Obviously, Acc resuts will not be relevant.",
    )

    parser.add_argument(
        "--data-loader",
        default="V1",
        type=str,
        help="'V1' or 'V2'. Last stable release with DataloaderV2 is 0.9.0.",
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
