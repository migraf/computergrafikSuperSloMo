import tensorflow as tf
from tensorpack import ModelDesc
from sklearn.impute import SimpleImputer
from tensorpack import logger
from dataflow import *
from tensorpack.tfutils.summary import add_moving_summary
import cv2




class FlowModel(ModelDesc):
    def __init__(self, name):
        self.name = name

    def inputs(self):
        return [tf.placeholder(tf.float32, (1,128,128,3), name="I_t_" + str(x)) for x in range(8)]


    def warping(self, img, flow):
        B = tf.shape(img)[0]
        c = tf.shape(img)[1]
        h = tf.shape(img)[2]
        w = tf.shape(img)[3]

        img_flat = tf.reshape(tf.transpose(img, [0,2,3,1]), [-1, c])
        dx,dy = tf.unstack(flow, axis=1)
        xf, yf = tf.meshgrid(tf.to_float(tf.range(w)), tf.to_float(tf.range(h)))
        xf = xf + dx
        yf = yf + dy

        alpha = tf.expand_dims(xf - tf.floor(xf), axis=1)
        beta = tf.expand_dims(yf - tf.floor(yf), axis=1)
        xL = tf.clip_by_value(tf.cast(tf.floor(xf), dtype=tf.int32), 0, w - 1)
        xR = tf.clip_by_value(tf.cast(tf.floor(xf) + 1, dtype=tf.int32), 0, w - 1)
        yT = tf.clip_by_value(tf.cast(tf.floor(yf), dtype=tf.int32), 0, h - 1)
        yB = tf.clip_by_value(tf.cast(tf.floor(yf) + 1, dtype=tf.int32), 0, h - 1)

        batch_ids = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(B), axis=-1), axis=-1), [1, h, w])

        def get(y, x):
            idx = tf.reshape(batch_ids * h * w + y * w + x, [-1])
            idx = tf.cast(idx, tf.int32)
            return tf.gather(img_flat, idx)

        val = tf.zeros_like(alpha)
        val += (1 - alpha) * (1 - beta) * tf.reshape(get(yT, xL), [-1, h, w, c])
        val += (0 + alpha) * (1 - beta) * tf.reshape(get(yT, xR), [-1, h, w, c])
        val += (1 - alpha) * (0 + beta) * tf.reshape(get(yB, xL), [-1, h, w, c])
        val += (0 + alpha) * (0 + beta) * tf.reshape(get(yB, xR), [-1, h, w, c])

        # we need to enforce the channel_dim known during compile-time here
        shp = img.shape.as_list()
        return tf.reshape(tf.transpose(val, [0, 3, 1, 2]), [-1, shp[1], h, w])


    def visualize_flow(self, flow):
        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx * fx + fy * fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = np.minimum(v * 4, 255)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return bgr


    def hierarchy_layer_down(self, input,  filter, kernel_size, skip_connections):
        """
        One Hierarchy Layer for the Encoder of the U-Net architecture
        :param input:
        :param filter: filter for convolution
        :param kernel_size: for convolution
        :return:
        """
        out = tf.layers.conv2d(input, filters = filter, kernel_size = kernel_size, strides=1, padding="same")
        out = tf.nn.leaky_relu(out, alpha=0.1)
        out = tf.layers.conv2d(out, filters = filter, kernel_size = kernel_size, strides = 1, padding="same")
        out = tf.nn.leaky_relu(out, alpha=0.1)

        skip_connections.append(out)

        out = tf.layers.average_pooling2d(out, 2, 2)
        return out

    def hierarchy_layer_up(self, input, skip_conection, filter):
        """
        Decoder Layer for U-Net architecture, doubles the size of the input
        and performs convolution
        :param input:
        :param skip_conection: output of previous layer for skip connection
        :param filter:
        :return:
        """
        sizes = input.shape
        # transform image to NHWC
        out = tf.image.resize_images(input, [sizes[1]*2, sizes[2]*2])
        # TODO change whole thing to one format either NHWC or NCHW
        # transform back to NCHW
        out = tf.layers.conv2d(out, filters=filter, kernel_size=3, strides=1,
                               padding="same")
        out = tf.nn.leaky_relu(out, alpha=0.1)
        out = tf.concat([out, skip_conection], axis=3)
        out = tf.layers.conv2d(out, filters=filter, kernel_size=3, strides = 1,
                               padding="same")
        out = tf.nn.leaky_relu(out, alpha=0.1)

        return out

    def basic_flow(self, I0, I1):
        """
        Base Flow compution network, predict the optical Flow F0_1 and F1_0
        between I0 and I1
        :param I0: image at time 0
        :param I1: image at time 1
        :return:
        """
        skip_connection = []
        input = tf.concat([I0, I1],axis=3)
        # U-Net Encoder
        # First Hierarchy Kernel Size 7
        out = self.hierarchy_layer_down(input, 32, 7, skip_connection)
        #skip_connection.append(out)
        # Second Hierarchy Kernel Size 5
        out = self.hierarchy_layer_down(out, 64, 5, skip_connection)
        #skip_connection.append(out)
        # Third Hierarchy layer
        out = self.hierarchy_layer_down(out, 128, 3, skip_connection)
        #skip_connection.append(out)
        # Fourth
        out = self.hierarchy_layer_down(out, 256, 3, skip_connection)
        #skip_connection.append(out)
        # Fifth
        out = self.hierarchy_layer_down(out, 512, 3, skip_connection)
        #skip_connection.append(out)
        # Sixth Layer no average pooling
        out = tf.layers.conv2d(out, filters = 512, kernel_size = 3, strides=1, padding="same")
        out = tf.nn.leaky_relu(out, alpha=0.1)
        out = tf.layers.conv2d(out, filters = 512, kernel_size = 3, strides = 1, padding="same")
        out = tf.nn.leaky_relu(out, alpha=0.1)



        # Decoder

        # 5 Hierarchies with skip connections to the encoder networks of the same size
        out = self.hierarchy_layer_up(out, skip_connection[-1], 512)
        out = self.hierarchy_layer_up(out, skip_connection[-2], 256)
        out = self.hierarchy_layer_up(out, skip_connection[-3], 128)
        out = self.hierarchy_layer_up(out, skip_connection[-4],  64)
        out = self.hierarchy_layer_up(out, skip_connection[-5],  32)

        # TODO can you just do this?

        out = tf.layers.conv2d(out, filters=4, kernel_size=3, strides=1, padding="same")
        out = tf.identity(out)
        print("Basic flow network created")

        return out

    def flow_interpolation(self, I_0, I_1, F_0_1, F_1_0, g_I1_F_t_1, g_I0_F_t_0, F_t_1, F_t_0):
        """
        U-Net architcture network for abritrary time flow interpolation
        :param I_0: Image at t = 0
        :param I_1: Image at t = 1
        :param F_0_1: Optical flow 0 -> 1 predicted by flow base flow network
        :param F_1_0: Optical flow 1 -> 0 predicted by flow base flow network
        :param g_I1_F_t_1: backwards warping function with I_1 and estimated flow from t -> 1
        :param g_I0_F_t_0: backwards warping function with I_0 and estimated flow from t -> 0
        :param F_t_1: Estimated flow from t -> 1
        :param F_t_0: Estimated flow from t -> 0
        :return:
        """
        skip_connections = []
        # concatenate inputs
        input = tf.concat([I_0, I_1, F_0_1, F_1_0, g_I1_F_t_1, g_I0_F_t_0, F_t_1, F_t_0], axis=3 )
        # same u-net architecture as base flow network
        # U-Net Encoder
        # First Hierarchy Kernel Size 7
        # size 512
        out = self.hierarchy_layer_down(input, 32, 7, skip_connections)
        #skip_connections.append(out)
        # Second Hierarchy Kernel Size 5
        # size 256
        out = self.hierarchy_layer_down(out, 64, 5, skip_connections)
        #skip_connections.append(out)
        # Third Hierarchy layer
        # size 128
        out = self.hierarchy_layer_down(out, 128, 3, skip_connections)
        #skip_connections.append(out)
        # Fourth
        # Size 64
        out = self.hierarchy_layer_down(out, 256, 3, skip_connections)
        # skip_connections.append(out)
        # Fifth
        # Size 32
        out = self.hierarchy_layer_down(out, 512, 3, skip_connections)
        #skip_connections.append(out)
        # Sixth Layer no average pooling
        out = tf.layers.conv2d(out, filters = 512, kernel_size = 3, strides=1, padding="same")
        out = tf.nn.leaky_relu(out, alpha=0.1)
        out = tf.layers.conv2d(out, filters = 512, kernel_size = 3, strides = 1, padding="same")
        out = tf.nn.leaky_relu(out, alpha=0.1)

        # Decoder

        # 5 Hierarchies with skip connections to the encoder networks of the same size
        out = self.hierarchy_layer_up(out, skip_connections[-1], 512)
        out = self.hierarchy_layer_up(out, skip_connections[-2], 256)
        out = self.hierarchy_layer_up(out, skip_connections[-3], 128)
        out = self.hierarchy_layer_up(out, skip_connections[-4],  64)
        out = self.hierarchy_layer_up(out, skip_connections[-5],  32)

        out = tf.identity(out)
        return out

    def simple_loss(self, reconstruction, frame):
        """
        Simple loss combination of L1 L2 and SSIM
        :param reconstruction:
        :param frame:
        :return:
        """
        l1 = tf.reduce_mean(tf.abs(tf.subtract(frame, reconstruction)))
        l2 = tf.reduce_mean(tf.squared_difference(frame, reconstruction))
        ssim = tf.squeeze(tf.image.ssim(frame, reconstruction, max_val=1.0))

        return l1 + l2 + ssim


    def reconstruction_loss(self, reconstruction, frame):
        return tf.reduce_mean(tf.abs(tf.subtract(frame, reconstruction)))

    def perceptual_loss(self, reconstructions, frames):
        # TODO load pretrained Image net VGG16 model and compute features, mabye do in build graph
        # TODO check for NCHW oder NHWC

        return 0
    def warping_loss(self, I0, I1, F_0_1, F_1_0 ):
        """
        Warping loss between two images
        TODO needs to be summed and normalized, how to use for intermediate frames?
        :param I0:
        :param I1:
        :param F_0_1:
        :param F_1_0:
        :return:
        """
        return 0

    def smoothness_loss(self, delta_F_0_1, delta_F_1_0):
        return tf.losses.absolute_difference(delta_F_0_1, delta_F_1_0)

    def build_graph(self, *args):
        loss = 0
        # Add summary

        intermediate_images = []
        basic_flow_result = self.basic_flow(args[0], args[-1])

        # Multiply flow by scalar because it is normalized between 0 and 1
        F_0_1 = tf.multiply(basic_flow_result[:, :, :, :2], 10, name="F_0_1")
        F_1_0 = tf.multiply(basic_flow_result[:, :, :, 2:], 10, name="F_1_0")

        tf.summary.image("Original Image 0", args[0], max_outputs=10)
        tf.summary.image("Original Image 1", args[-1], max_outputs=10)

        # loss for computed flow t0 -> t1 and t1 -> t0
        warped_image_0 = tf.contrib.image.dense_image_warp(args[-1], F_1_0)
        warped_image_1 = tf.contrib.image.dense_image_warp(args[0], F_0_1)


        loss += self.simple_loss(args[-1],warped_image_1)
        loss += self.simple_loss(args[0], warped_image_0)

        warped_image_0_scaled = tf.multiply(warped_image_0, 255)
        warped_image_1_scaled = tf.multiply(warped_image_1, 255)

        tf.summary.image("Warped Image t0", warped_image_0_scaled, max_outputs=10 )
        tf.summary.image("warped Image t1", warped_image_1_scaled, max_outputs=10)

        print(type(F_0_1))

        #flow_viz_0_1 = self.visualize_flow(F_0_1.eval())
        #flow_viz_1_0 = self.visualize_flow(F_1_0.eval())

        #viz = tf.concat([args[0], flow_viz_0_1, flow_viz_1_0, args[-1]], 2)

        #tf.summary.image("Flow visualization", viz, max_outputs=10)

        all_frames = tf.concat([args[i] for i in range(len(args))], axis=2)

        tf.summary.image("All consecutive frames", all_frames, max_outputs=10)

        tf.summary.scalar("loss", loss)
        print(loss)


        interpolated_frames = []
        with tf.name_scope("loss_intermediate_frames"):
            # Iterate over intermediate frames
            for it in range(1,8):
                t = it/8
                F_t_0 = -(1 - t)*t*F_0_1 + t ** 2 * F_1_0
                F_t_1 = (1 - t)**2 *F_0_1 - t * (1- t) * F_1_0

                g_I0_F_t_0 = tf.contrib.image.dense_image_warp(args[0], F_t_0)
                g_I1_F_t_0 = tf.contrib.image.dense_image_warp(args[-1], F_t_1)

                interpolation_result = self.flow_interpolation(args[0], args[-1], F_0_1, F_1_0 , g_I1_F_t_0,
                                                               g_I0_F_t_0, F_t_1, F_t_0 )
                # get results for visibility maps from interpolation result
                F_t_0_net = interpolation_result[:,:,:,:2] + F_t_0
                F_t_1_net = interpolation_result[:,:,:,2:4] + F_t_1
                V_t_0 = tf.expand_dims(interpolation_result[:,:,:,5], axis=3)
                V_t_1 = 1 - V_t_0

                g_I0_F_t_0_net = tf.contrib.image.dense_image_warp(args[0], F_t_0_net)
                g_I1_F_t_0_net = tf.contrib.image.dense_image_warp(args[-1], F_t_1_net)

                # normalization for visibility fields
                norm_vis = (1- t) * V_t_0 + t*V_t_1

                # calculate interpolated image and normalize
                interpolated_image = (1-t)*V_t_0*g_I0_F_t_0_net + t * V_t_1 * g_I1_F_t_0_net
                interpolated_image = tf.divide(interpolated_image,norm_vis, name="interpolated_image")
                print(type(interpolated_image))

                # add to list for visualization
                interpolated_frames.append(interpolated_image)

                # compute loss for intermediate image
                loss += self.simple_loss(interpolated_image, args[it])

        tf.summary.image("Interpolated consecutive frames", tf.concat(interpolated_frames, axis=2), max_outputs=10)

        self.cost = loss
        add_moving_summary(self.cost)

        return loss


    def optimizer(self):
        return tf.train.AdamOptimizer(0.0001)



