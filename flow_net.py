import tensorflow as tf
from tensorpack import ModelDesc
from tensorpack import logger
from dataflow import *
from tensorpack.tfutils.summary import add_moving_summary
import cv2


class FlowNetModel(ModelDesc):
    def __init__(self, name, height, width):
        self.name = name
        self.height = height
        self.width = width

    def input_names(self):
        return [tf.placeholder(tf.float32, (1,3, self.height, self.width), name="left_image"),
                tf.placeholder(tf.float32, (1,3,self.height, self.width), name="right_image"),
                tf.placeholder(tf.float32, (1,2, self.height, self.width), name="flow")]

    def correlation(ina, inb,
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
                ans = tf.reduce_mean(ina * s, axis=1, keepdims=True)
                res.append(ans)
        res = tf.concat(res, axis=3)  # ND^2HW
        return res

    def _build_graph(self, left_image, right_image, flow):

        # Left channel of correlated flow net architecture, figure2 in Paper
        left_channel = tf.layers.conv2d(left_image, 64, kernel_size=7, strides=(2,2),
                                        activation=tf.nn.relu, name="left_conv0", padding="same")
        left_conv1 = tf.layers.conv2d(left_channel, 128, kernel_size=5, strides=(2,2),
                                        activation=tf.nn.relu, name="left_conv1", padding="same")
        left_conv2 = tf.layers.conv2d(left_conv1, 256, kernel_size=5, strides=(2,2),
                                        activation=tf.nn.relu, name="left_conv2", padding="same")

        # Right channel of correlated flow net architecture, figure2 in Paper
        right_channel = tf.layers.conv2d(right_image, 64, kernel_size=7, strides=(2,2),
                                        activation=tf.nn.relu, name="right_conv0", padding="same")
        right_conv1 = tf.layers.conv2d(right_channel, 128, kernel_size=5, strides=(2,2),
                                        activation=tf.nn.relu, name="right_conv1", padding="same")
        right_conv2 = tf.layers.conv2d(right_conv1, 256, kernel_size=5, strides=(2,2),
                                        activation=tf.nn.relu, name="right_conv2", padding="same")

        corr = self.correlation(left_conv2, right_conv2,
                           kernel_size=1,
                           max_displacement=20,
                           stride_1=1,
                           stride_2=2,
                           pad=20, data_format='NCHW')

        corr = tf.nn.relu(corr)

        left_conv_input = tf.layers.conv2d(left_image, 32, kernel_size=1, strides=(1,1),
                                        activation=tf.nn.relu, name="left_conv_input", padding="same")

        # Contracting Part of the architecture

        corr_conc = tf.concat([corr, left_conv_input], axis=1)

        conv_3_1 = tf.layers.conv2d(corr_conc, 256, kernel_size=3, strides=(1,1), padding="same",
                                    activation=tf.nn.relu, name="conv_3_1")
        conv4 = tf.layers.conv2d(conv_3_1, 512, kernel_size=3, strides=(1,1), padding="same",
                                    activation=tf.nn.relu, name="conv_4")

        conv_4_1 = tf.layers.conv2d(conv4, 512, kernel_size=3, strides=(1, 1), padding="same",
                         activation=tf.nn.relu, name="conv_4_1")

        conv5 = tf.layers.conv2d(conv_4_1, 512, kernel_size=3, strides=(1,1), padding="same",
                                    activation=tf.nn.relu, name="conv_5")

        conv_5_1 = tf.layers.conv2d(conv5, 512, kernel_size=3, strides=(1,1), padding="same",
                                    activation=tf.nn.relu, name="conv_5_1")

        conv6 = tf.layers.conv2d(conv_5_1, 1024, kernel_size=3, strides=(1,1), padding="same",
                                    activation=tf.nn.relu, name="conv_4_0")

        # Extracting Part of the architecture

        upconv5 = tf.layers.conv2d_transpose(conv6, 512, kernel_size=5, strides=(2,2), padding="same",
                                             activation=tf.nn.relu, name="upconv5")
        concat = tf.concat([upconv5, conv_5_1], axis=1)
        predict_flow5 = tf.layers.conv2d(concat, 2, kernel_size=5, strides=(2,2), padding="same",
                                         activation=tf.identity, name="flow5")

        # Second Flow prediction

        upconv4 = tf.layers.conv2d_transpose(concat, 256, kernel_size=1, strides=(1,1), padding="same",
                                             activation=tf.nn.relu, name="upconv4")
        concat = tf.concat([upconv4, conv_4_1, predict_flow5], axis=1)
        predict_flow4 = tf.layers.conv2d(concat, 2, kernel_size=5, strides=(2,2), padding="same",
                                         activation=tf.identity, name="flow4")

        # Third Flow

        upconv3 = tf.layers.conv2d_transpose(concat, 128, kernel_size=1, strides=(1,1), padding="same",
                                             activation=tf.nn.relu, name="upconv3")
        concat = tf.concat([upconv3, conv_3_1, predict_flow4], axis=1)
        predict_flow3 = tf.layers.conv2d(concat, 2, kernel_size=5, strides=(2,2), padding="same",
                                         activation=tf.identity, name="flow3")

        # Final Flow

        upconv2 = tf.layers.conv2d_transpose(concat, 64, kernel_size=1, strides=(1,1), padding="same",
                                             activation=tf.nn.relu, name="upconv2")
        concat = tf.concat([upconv2, conv_5_1, predict_flow4], axis=1)
        final_flow = tf.layers.conv2d(concat, 2, kernel_size=5, strides=(2,2), padding="same",
                                         activation=tf.identity, name="flow3")

        # Use nearest neighbur upsampling to get the correct shape upsampling on output

        new_shape = final_flow.shape[1:3] * 4
        upsampled_flow = tf.image.resize_nearest_neighbor(tf.multiply(final_flow, 20) ,  new_shape)

        final_prediction = tf.identity(upsampled_flow, name="final_prediction")

        epe = tf.reduce_mean(tf.norm(final_prediction - flow, axis=3))
        add_moving_summary(epe)

        self.cost = epe

        return self.cost

















