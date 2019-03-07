from tensorpack import *
from tensorpack.utils import logger
from tensorpack.train import TrainConfig
from tensorpack.train import TowerTrainer
from tensorpack.train import SingleCostTrainer
import dataflow
import models
import argparse
from tensorpack.dataflow.serialize import LMDBSerializer
from callbacks import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="directory containing the training images")
    parser.add_argument("--lmdb_path", help="Path of the lmdb file", default="/graphics/scratch/students/graf/data/all.lmdb")
    parser.add_argument("--image_size", help="size to scale the images to", default=512)
    parser.add_argument("--num_frames", help="number of intermediate frames", default=8)
    parser.add_argument("--num_batches", default=1)
    parser.add_argument("--gpus" ,help="comma separated list of gpus to use", default="1,2,3")
    # TODO what else do we need

    args = parser.parse_args()

    logger.auto_set_dir()
    if args.lmdb_path:
        df = LMDBSerializer.load(args.lmdb_path, shuffle=False)
    else:
        df = dataflow.IntermediateDataFlow(args.file_path, args.num_frames, args.image_size)
    #df = PrefetchData(df, 2,2)
    df = BatchData(df, int(args.num_batches))

    model = models.FlowModel("FlowModel", int(args.num_batches))
    # TODO is this needed/ just use defaults?
    config = TrainConfig(
        model=model,
        dataflow=df,
        max_epoch=15,
        callbacks= [ModelSaver(), FlowVisualisationCallback(["F_0_1", "F_1_0"]),
                    ],
        steps_per_epoch=df.size(),
        nr_tower=len(args.gpus.split(','))
    )
    trainer = SyncMultiGPUTrainer(config.nr_tower)
    launch_train_with_config(config, trainer)