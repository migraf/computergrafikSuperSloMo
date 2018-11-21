import tensorflow as tf
from tensorpack import ModelDesc
from sklearn.impute import SimpleImputer
from tensorpack import logger
from dataflow import *
from tensorpack.tfutils.summary import add_moving_summary




class FlowModel(ModelDesc):
    def __init__(self, name):
        self.name = name

    def inputs(self, num_intermediate_frames):
        return [tf.placeholder(tf.float32, (360,360), name="It_" + x) for x in range(num_intermediate_frames)]

    # TODO write backwards warping function with interpolations

    def warping(self, img, flow):
        N = tf.shape(img)[0]
        H = tf.shape(img)[1]
        W = tf.shape(img)[2]
        C = tf.shape(img)[3]
        # TODO is transpose really necessary why is this done?
        img_flat = tf.reshape(tf.transpose(img, [0,2,3,1]), [-1, C])
        dx,dy = tf.unstack(flow, axis=1)
        xf, yf = tf.meshgrid(tf.to_float(tf.range(W)), tf.to_float(tf.range(H)))
        xf = xf + dx
        yf = yf + dy

        alpha = tf.expand_dims(xf - tf.floor(xf), axis=1)
        beta = tf.expand_dims(yf - tf.floor(yf), axis=1)




    def hierarchy_layer_down(self, input,  filter, kernel_size):
        """
        One Hierarchy Layer for the Encoder of the U-Net architecture
        :param input:
        :param filter: filter for convolution
        :param kernel_size: for convolution
        :return:
        """
        # TODO pad inputs?
        out = tf.layers.conv2d(input, filters = filter, kernel_size = kernel_size, strides=1)
        out = tf.nn.leaky_relu(out, alpha=0.1)
        out = tf.layers.conv2d(out, filters = filter, kernel_size = kernel_size, strides = 1)
        out = tf.nn.leaky_relu(out, alpha=0.1)
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
        out = tf.image.resize_bilinear(input, [sizes[1]*2, sizes[2]*2])
        # TODO is this the correct way for skip connections, maybe concat
        out = input + skip_conection
        out = tf.layers.conv2d(out, filters=filter, kernel_size=3, strides=1)
        out = tf.nn.leaky_relu(out, alpha=0.1)
        out = tf.layers.conv2d(out, filters=filter, kernel_size=3, strides = 1)
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
        input = tf.concat(I0, I1,axis=1)

        # U-Net Encoder

        # First Hierarchy Kernel Size 7
        out = self.hierarchy_layer_down(input, 32, 7)
        skip_connection.append(out)
        # Second Hierarchy Kernel Size 5
        out = self.hierarchy_layer_down(out, 64, 5)
        skip_connection.append(out)
        # Third Hierarchy layer
        out = self.hierarchy_layer_down(out, 128, 3)
        skip_connection.append(out)
        # Fourth
        out = self.hierarchy_layer_down(out, 256, 3)
        skip_connection.append(out)
        # Fifth
        out = self.hierarchy_layer_down(out, 512, 3)
        skip_connection.append(out)
        # Sixth Layer no average pooling
        out = tf.layers.conv2d(input, filters = 512, kernel_size = 3, strides=1)
        out = tf.nn.leaky_relu(out, alpha=0.1)
        out = tf.layers.conv2d(out, filters = 512, kernel_size = 3, strides = 1)
        out = tf.nn.leaky_relu(out, alpha=0.1)

        # Decoder

        # 5 Hierarchies with skip connections to the encoder networks of the same size
        out = self.hierarchy_layer_up(out, skip_connection[-1], 512)
        out = self.hierarchy_layer_up(out, skip_connection[-2], 256)
        out = self.hierarchy_layer_up(out, skip_connection[-3], 128)
        out = self.hierarchy_layer_up(out, skip_connection[-4],  64)
        out = self.hierarchy_layer_up(out, skip_connection[-5],  32)

        # TODO how to know if the output has the right dimensions?
        print(tf.shape(out))

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

        input = tf.concat([I_0, I_1, F_0_1, F_1_0, g_I1_F_t_1, g_I0_F_t_0, F_t_1, F_t_0], axis=1)

        # same u-net architecture as base flow network
        # U-Net Encoder

        # First Hierarchy Kernel Size 7
        out = self.hierarchy_layer_down(input, 32, 7)
        skip_connections.append(out)
        # Second Hierarchy Kernel Size 5
        out = self.hierarchy_layer_down(out, 64, 5)
        skip_connections.append(out)
        # Third Hierarchy layer
        out = self.hierarchy_layer_down(out, 128, 3)
        skip_connections.append(out)
        # Fourth
        out = self.hierarchy_layer_down(out, 256, 3)
        skip_connections.append(out)
        # Fifth
        out = self.hierarchy_layer_down(out, 512, 3)
        skip_connections.append(out)
        # Sixth Layer no average pooling
        out = tf.layers.conv2d(input, filters = 512, kernel_size = 3, strides=1)
        out = tf.nn.leaky_relu(out, alpha=0.1)
        out = tf.layers.conv2d(out, filters = 512, kernel_size = 3, strides = 1)
        out = tf.nn.leaky_relu(out, alpha=0.1)

        # Decoder

        # 5 Hierarchies with skip connections to the encoder networks of the same size
        out = self.hierarchy_layer_up(out, skip_connections[-1], 512)
        out = self.hierarchy_layer_up(out, skip_connections[-2], 256)
        out = self.hierarchy_layer_up(out, skip_connections[-3], 128)
        out = self.hierarchy_layer_up(out, skip_connections[-4],  64)
        out = self.hierarchy_layer_up(out, skip_connections[-5],  32)

        #TODO check dimensions
        print(tf.shape(out))
        return out

    def simple_loss(self, reconstruction, frame):
        """
        Simple loss combination of L1 L2 and SSIM
        :param reconstruction:
        :param frame:
        :return:
        """
        l1 = tf.losses.absolute_difference(frame, reconstruction)
        l2 = tf.losses.mean_squared_error(frame, reconstruction)
        ssim = tf.image.ssim(frame, reconstruction, max_val=1.0)

        return l1 + l2 + ssim


    def reconstruction_loss(self, reconstruction, frame):
        return tf.losses.absolute_difference(frame, reconstruction)

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
        # TODO compute the loss functions
        # TODO how to handle intermediate frames?
        loss = 0
        # Add summary
        add_moving_summary(loss)
        intermediate_images = []
        basic_flow_result = self.basic_flow(args[0], args[-1])
        # TODO is this the right way to get the flow from the basic flow net, indexes correct?
        F_0_1 = basic_flow_result[:, :2, :, :]
        F_1_0 = basic_flow_result[:, 2:, :, :]
        with tf.name_scope("loss basic flow"):
            # loss for computed flow t0 -> t1 and t1 -> t0

            loss += self.simple_loss(args[-1],self.warping(args[0], F_0_1))
            loss += self.simple_loss(args[0], self.warping(args[-1], F_1_0))


        with tf.name_scope("loss intermediate frames"):
            # Iterate over intermediate frames
            for it in range(1,8):
                t = it/8
                F_t_0 = -(1 - t)*t*F_0_1 + t ** 2 * F_1_0
                F_t_1 = (1 - t)**2 *F_0_1 - t * (1- t) * F_1_0

                g_I0_F_t_0 = self.warping(args[0], F_t_0)
                g_I1_F_t_0 = self.warping(args[-1], F_t_1)

                interpolation_result = self.flow_interpolation(args[0], args[-1], F_0_1, F_1_0 , g_I1_F_t_0,
                                                               g_I0_F_t_0, F_t_1, F_t_0 )
                print(tf.shape(interpolation_result))

                # get results for visibility maps from interpolation result
                F_t_0_net = interpolation_result[:,:2,:,:] + F_t_0
                F_t_1_net = interpolation_result[:,2:4,:,:] + F_t_1
                V_t_0 = tf.expand_dims(interpolation_result[:,4:,:,:], axis=1)
                V_t_1 = 1 - V_t_0

                g_I0_F_t_0_net = self.warping(args[0], F_t_0_net)
                g_I1_F_t_0_net = self.warping(args[-1], F_t_1_net)

                # normalization for visibility fields
                norm_vis = (1- t) * V_t_0 + t*V_t_1

                # calculate interpolated image and normalize
                interpolated_image = (1-t)*V_t_0*g_I0_F_t_0_net + t * V_t_1 * g_I1_F_t_0_net
                interpolated_image = interpolated_image/norm_vis

                # compute loss for intermediate image
                loss += self.simple_loss(interpolated_image, args[it])

        return loss


    def optimizer(self):
        return tf.train.AdamOptimizer(0.0001)



