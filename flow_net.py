import tensorflow as tf
from tensorpack import ModelDesc
from tensorpack import logger
from dataflow import *
from tensorpack.tfutils.summary import add_moving_summary
import cv2
from flownet_util import *
from tensorpack.train import TrainConfig
from tensorpack import *


class FlowNetModel(ModelDesc):
    def __init__(self, name, height, width, num_batches):
        self.name = name
        self.height = height
        self.width = width
        self.num_batches = num_batches
        self.lr = tf.get_variable("lr", trainable=False, initializer=1e-6)

    def inputs(self):
        return [tf.placeholder(tf.float32, (self.num_batches, self.height, self.width, 3), name="left_image"),
                tf.placeholder(tf.float32, (self.num_batches, self.height, self.width, 3), name="right_image"),
                tf.placeholder(tf.float32, (self.num_batches, self.height, self.width, 2), name="flow")]

    def correlation(self, ina, inb,
                    kernel_size, max_displacement,
                    stride_1, stride_2,
                    pad, data_format):
        """
        Correlation Cost Volume computation.
        This is a fallback Python-only implementation, specialized just for FlowNet2.
        It takes a lot of memory and is slow.
        If you know to compile a custom op yourself, it's better to use the cuda implementation here:
        https://github.com/PatWie/tensorflow-recipes/tree/master/OpticalFlow/user_ops
        """
        assert pad == max_displacement
        assert kernel_size == 1
        assert data_format == 'NHWC'
        assert max_displacement % stride_2 == 0
        assert stride_1 == 1

        D = int(max_displacement / stride_2 * 2) + 1  # D^2 == number of correlations per spatial location

        b, h, w, c = ina.shape.as_list()

        inb = tf.pad(inb, [[0, 0], [pad, pad], [pad, pad], [0, 0]])

        res = []
        for k1 in range(0, D):
            start_h = k1 * stride_2
            for k2 in range(0, D):
                start_w = k2 * stride_2
                s = tf.slice(inb, [0, start_h, start_w, 0], [-1, h, w, -1])
                ans = tf.reduce_mean(ina * s, axis=3, keepdims=True)
                res.append(ans)
        res = tf.concat(res, axis=3)  # ND^2HW
        return res

    def build_graph(self, *args):
        print(args[2].shape)
        print(args)

        # Left channel of correlated flow net architecture, figure2 in Paper
        left_channel = tf.layers.conv2d(tf.pad(args[0], [[0, 0], [3, 3], [3, 3], [0, 0]]), 64, kernel_size=7,
                                        strides=(2, 2),
                                        activation=tf.nn.relu, name="left_conv0", padding="valid")
        left_conv1 = tf.layers.conv2d(tf.pad(left_channel, [[0, 0], [2, 2], [2, 2], [0, 0]]), 128, kernel_size=5,
                                      strides=(2, 2),
                                      activation=tf.nn.relu, name="left_conv1", padding="valid")
        left_conv2 = tf.layers.conv2d(tf.pad(left_conv1, [[0, 0], [2, 2], [2, 2], [0, 0]]), 256, kernel_size=5,
                                      strides=(2, 2),
                                      activation=tf.nn.relu, name="left_conv2", padding="valid")

        # Right channel of correlated flow net architecture, figure2 in Paper
        right_channel = tf.layers.conv2d(tf.pad(args[1], [[0, 0], [3, 3], [3, 3], [0, 0]]), 64, kernel_size=7,
                                         strides=(2, 2),
                                         activation=tf.nn.relu, name="right_conv0", padding="valid")
        right_conv1 = tf.layers.conv2d(tf.pad(right_channel, [[0, 0], [2, 2], [2, 2], [0, 0]]), 128, kernel_size=5,
                                       strides=(2, 2),
                                       activation=tf.nn.relu, name="right_conv1", padding="valid")
        right_conv2 = tf.layers.conv2d(tf.pad(right_conv1, [[0, 0], [2, 2], [2, 2], [0, 0]]), 256, kernel_size=5,
                                       strides=(2, 2),
                                       activation=tf.nn.relu, name="right_conv2", padding="valid")

        corr = self.correlation(left_conv2, right_conv2, 1, 20, 1, 2, 20, "NHWC")

        corr = tf.nn.relu(corr)

        left_conv_input = tf.layers.conv2d(left_conv2, 32, kernel_size=1, strides=(1, 1),
                                           activation=tf.nn.relu, name="left_conv_input", padding="valid")

        # Contracting Part of the architecture

        corr_conc = tf.concat([corr, left_conv_input], axis=3)

        conv_3_1 = tf.layers.conv2d(tf.pad(corr_conc, [[0, 0], [1, 1], [1, 1], [0, 0]]), 256, kernel_size=3,
                                    strides=(1, 1), padding="valid",
                                    activation=tf.nn.relu, name="conv_3_1")

        print("Conv 3_1 shape: {}".format(conv_3_1.shape))

        conv4 = tf.layers.conv2d(tf.pad(conv_3_1, [[0, 0], [1, 1], [1, 1], [0, 0]]), 512, kernel_size=3, strides=(2, 2),
                                 padding="valid",
                                 activation=tf.nn.relu, name="conv_4")

        print("Conv 4 shape: {}".format(conv4.shape))

        conv_4_1 = tf.layers.conv2d(tf.pad(conv4, [[0, 0], [1, 1], [1, 1], [0, 0]]), 512, kernel_size=3, strides=(1, 1),
                                    padding="valid",
                                    activation=tf.nn.relu, name="conv_4_1")
        print("Conv 4_1 shape: {}".format(conv_4_1.shape))

        conv5 = tf.layers.conv2d(tf.pad(conv_4_1, [[0, 0], [1, 1], [1, 1], [0, 0]]), 512, kernel_size=3, strides=(2, 2),
                                 padding="valid",
                                 activation=tf.nn.relu, name="conv_5")
        print("Conv 5 shape: {}".format(conv5.shape))
        conv_5_1 = tf.layers.conv2d(tf.pad(conv5, [[0, 0], [1, 1], [1, 1], [0, 0]]), 512, kernel_size=3, strides=(1, 1),
                                    padding="valid",
                                    activation=tf.nn.relu, name="conv_5_1")
        print("Conv 5_1 shape: {}".format(conv_5_1.shape))
        conv6 = tf.layers.conv2d(tf.pad(conv_5_1, [[0, 0], [1, 1], [1, 1], [0, 0]]), 1024, kernel_size=3,
                                 strides=(2, 2), padding="valid",
                                 activation=tf.nn.relu, name="conv_4_0")

        print("Conv 6 shape: {}".format(conv6.shape))

        # Extracting Part of the architecture

        upconv5 = tf.layers.conv2d_transpose(conv6, 512, kernel_size=4, strides=(2, 2), padding="same",
                                             activation=tf.nn.relu, name="upconv5")

        print("UpConv 5 shape: {}".format(upconv5.shape))

        concat = tf.concat([upconv5, conv_5_1], axis=3)

        predict_flow5 = tf.layers.conv2d(tf.pad(concat, [[0, 0], [2, 2], [2, 2], [0, 0]]), 2, kernel_size=5,
                                         strides=(1, 1), padding="valid",
                                         activation=tf.identity, name="flow5")
        flow_5_up = tf.layers.conv2d_transpose(predict_flow5, 2, kernel_size=4, strides=(2, 2), padding="same",
                                               activation=tf.identity)

        print("Predict flow shape:")
        print(predict_flow5.shape)
        print("Upflow shape")
        print(flow_5_up.shape)
        # tf.summary.image(name="flow5", tensor=visualize_flow(predict_flow5), max_outputs=3)

        # Second Flow prediction

        upconv4 = tf.layers.conv2d_transpose(concat, 256, kernel_size=4, strides=(2, 2), padding="same",
                                             activation=tf.nn.relu, name="upconv4")
        print("Upconv4 shape")
        print(upconv4.shape)

        concat = tf.concat([upconv4, conv_4_1, flow_5_up], axis=3)
        predict_flow4 = tf.layers.conv2d(tf.pad(concat, [[0, 0], [2, 2], [2, 2], [0, 0]]), 2, kernel_size=5,
                                         strides=(1, 1), padding="valid",
                                         activation=tf.identity, name="flow4")

        print("predictflow 4 shape: {}".format(predict_flow4.shape))
        flow_4_up = tf.layers.conv2d_transpose(predict_flow4, 2, kernel_size=4, strides=(2, 2), padding="same",
                                               activation=tf.identity, name="flow_4_up")
        # tf.summary.image(name="flow4", tensor=visualize_flow(predict_flow4), max_outputs=3)
        print("flow 4 up shape: {}".format(flow_4_up.shape))

        # Third Flow

        upconv3 = tf.layers.conv2d_transpose(concat, 128, kernel_size=5, strides=(2, 2), padding="same",
                                             activation=tf.nn.relu, name="upconv3")

        print("UpConv 3 shape: {}".format(upconv3.shape))

        concat = tf.concat([upconv3, conv_3_1, flow_4_up], axis=3)

        print("Concat shape: {}".format(concat.shape))

        predict_flow3 = tf.layers.conv2d(tf.pad(concat, [[0, 0], [2, 2], [2, 2], [0, 0]]), 2, kernel_size=5,
                                         strides=(1, 1), padding="valid",
                                         activation=tf.identity, name="flow3")
        print("Predict flow 3 shape: {}".format(predict_flow3.shape))

        flow_3_up = tf.layers.conv2d_transpose(predict_flow3, 2, kernel_size=4, strides=(2, 2), padding="same",
                                               activation=tf.identity, name="flow_3_up")

        print("Flow 3 up shape: {}".format(flow_3_up.shape))
        # tf.summary.image(name="flow3", tensor=visualize_flow(predict_flow3), max_outputs=3)

        # Final Flow
        upconv2 = tf.layers.conv2d_transpose(concat, 64, kernel_size=4, strides=(2, 2), padding="same",
                                             activation=tf.nn.relu, name="upconv2")
        concat = tf.concat([upconv2, left_conv1, flow_3_up], axis=3)
        final_flow = tf.layers.conv2d(tf.pad(concat, [[0, 0], [2, 2], [2, 2], [0, 0]]), 2, kernel_size=5,
                                      strides=(2, 2), padding="valid",
                                      activation=tf.identity, name="final_flow")

        # Use nearest neighbur upsampling to get the correct shape upsampling on output

        print("Final flow shape: {}".format(final_flow.shape))

        new_shape = (int(final_flow.shape[1]) * 8, int(final_flow.shape[2]) * 8)
        print("Final shape: {}".format(new_shape))

        upsampled_flow = tf.image.resize_nearest_neighbor(tf.multiply(final_flow, 20), new_shape)

        print("Upsampled flow shape: {}".format(upsampled_flow.shape))

        # Flows to visualize in tensorboard
        final_prediction = tf.identity(upsampled_flow, name="final_prediction")
        gt_flow = tf.identity(args[2], name="gt_flow")

        # Outer Images visualized in tensorboard
        tf.summary.image("left_image", args[0])
        tf.summary.image("right_image", args[1])

        # images_flow = tf.concat([args[0], visualize_flow(final_prediction), args[1]], axis=3)

        # tf.summary.image(images_flow, max_outputs=10)

        # tf.summary.image(name="flow_prediction", tensor=visualize_flow(final_prediction), max_outputs=3)

        epe = tf.reduce_mean(tf.norm(final_prediction - gt_flow, axis=3))
        print("Error shape: {}".format(epe.shape))
        add_moving_summary(epe)

        self.cost = epe

        return self.cost

    def optimizer(self):
        return tf.train.AdamOptimizer(self.lr)


def inference(saved_model, left_image_path, right_image_path, gt_flow=None):
    print(left_image_path)
    print(right_image_path)
    print(gt_flow)
    left_image = np.expand_dims(cv2.imread(left_image_path), 0)
    right_image = np.expand_dims(cv2.imread(right_image_path), 0)

    height = left_image.shape[1]
    width = left_image.shape[2]

    predict_config = PredictConfig(
        model=FlowNetModel("test_model", height, width, 1),
        session_init=get_model_loader(saved_model),
        input_names=["left_image", "right_image"],
        output_names=["final_prediction"])

    predictor = OfflinePredictor(predict_config)

    flow_prediction = predictor(left_image, right_image)
    flow_viz = visualize_flow(flow_prediction[0])
    print("max flow value: {}".format(np.max(flow_viz)))
    print(np.mean(np.linalg.norm(flow_prediction - read_flow(gt_flow), axis=3, ord=2)))
    print("Epe error: {}".format(tf.reduce_mean(tf.norm(flow_prediction - read_flow(gt_flow), axis=3))))

    # cv2.imshow("flow", flow_viz)
    cv2.imwrite('flow.png', flow_viz * 255)

    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="array containing the training images",
                        default="/graphics/scratch/students/graf/computergrafikSuperSloMo/train_paths.npy")
    parser.add_argument("--num_batches", default=1)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--test")

    args = parser.parse_args()

    if args.test:
        test_paths = np.load("/graphics/scratch/students/graf/computergrafikSuperSloMo/test_paths.npy")
        model_path = "/graphics/scratch/students/graf/computergrafikSuperSloMo/train_log/flow_net01/model-91485.data-00000-of-00001"

        inference(model_path, test_paths[0, 0], test_paths[1, 0], test_paths[2, 0])

    else:
        logger.auto_set_dir()

        df1 = FlownetDataflow(args.file_path)
        df = BatchData(df1, int(args.num_batches))

        # Steps at which to increase the learning rate
        lr_increase_schedule = [(10000, 1e-4), (300000, (1e-4) / 2), (400000, (1e-4) / 4), (500000, (1e-4) / 8)]

        model = FlowNetModel("flownet", df1.height, df1.width, int(args.num_batches))
        # df = QueueInput(df)
        # df = StagingInput(df, nr_stage=1)

        config = TrainConfig(
            model=model,
            dataflow=df,
            max_epoch=40,
            callbacks=[ModelSaver(),
                       PeriodicCallback(FlowVisualisationCallback(["final_prediction", "gt_flow"]),
                                        every_k_steps=10000),
                       ScheduledHyperParamSetter("lr", lr_increase_schedule, step_based=True)
                       ],
            steps_per_epoch=df1.size(),
        )
        trainer = SyncMultiGPUTrainer(1)
        launch_train_with_config(config, trainer)
