from tensorpack import *
from tensorpack.utils import logger
from tensorpack.train import TrainConfig
from tensorpack.train import TowerTrainer
from tensorpack.train import SingleCostTrainer
import dataflow
import models
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="directory containing the training images")
    parser.add_argument("--image_size", help="size to scale the images to")
    parser.add_argument("--num_frames", help="number of intermediate frames")
    # TODO what else do we need

    args = parser.parse_args()

    logger.auto_set_dir()

    df = dataflow.IntermediateDataFlow(args.file_path, args.num_frames, args.image_size)
    model = models.FlowModel()
    # TODO is this needed/ just use defaults?
    config = TrainConfig(
        model=model,
        dataflow=df,
        max_epoch=10,
        steps_per_epoch=df.size()

    )
    trainer = TowerTrainer()
    launch_train_with_config(config, trainer)