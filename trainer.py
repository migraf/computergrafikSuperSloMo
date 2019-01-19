from tensorpack import *
from tensorpack.utils import logger
from tensorpack.train import TrainConfig
from tensorpack.train import TowerTrainer
from tensorpack.train import SingleCostTrainer
import dataflow
import models
import argparse
from tensorpack.dataflow.serialize import LMDBSerializer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="directory containing the training images")
    parser.add_argument("--lmdb_path", help="Path of the lmdb file")
    parser.add_argument("--image_size", help="size to scale the images to", default=512)
    parser.add_argument("--num_frames", help="number of intermediate frames", default=8)
    # TODO what else do we need

    args = parser.parse_args()

    logger.auto_set_dir()
    if args.lmdb_path:
        df = LMDBSerializer.load(args.lmdb_path, shuffle=False)
    else:
        df = dataflow.IntermediateDataFlow(args.file_path, args.num_frames, args.image_size)
    print("Dataframe size")
    print(df.size())
    print(df.__len__())
    model = models.FlowModel("FlowModel")
    # TODO is this needed/ just use defaults?
    config = TrainConfig(
        model=model,
        dataflow=df,
        max_epoch=10,
        callbacks= [ModelSaver(),],
        steps_per_epoch=df.size()
    )
    trainer = SimpleTrainer()
    launch_train_with_config(config, trainer)