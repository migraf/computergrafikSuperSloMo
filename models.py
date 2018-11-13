import tensorflow as tf
from tensorpack import ModelDesc





class FlowModel(ModelDesc):
    # TODO what needs to be done at initialization?
    def __init__(self, name):
        self.name = name

    def inputs(self, num_intermediate_frames):
        # TODO increase number of placeholders
        return [tf.placeholder(tf.float32, (360,360), name="I" + x) for x in range(num_intermediate_frames)]

    # TODO write backwards warping function with interpolations

    def hierarchy_layer_down(self, input,  filter, kernel_size):
        """
        One Hierarchy Layer for the Encoder of the U-Net architecture
        :param input:
        :param filter: filter for convolution
        :param kernel_size: for convolution
        :return:
        """
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
        out = tf.image.resize_bilinear(out, [sizes[1]*2, sizes[2]*2])
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

        return out


    def flow_intepolation(self, I_0, I_1, F_0_1, F_1_0, g_I1_F_t_1, g_I0_F_t_0, F_t_1, F_t_0):
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

        return out

    def reconstruction_loss(self, reconstruction, frame):
        return tf.losses.absolute_difference(frame, reconstruction)

    def perceptual_loss(self, reconstructions, frames):
        # TODO load pretrained Image net VGG16 model and compute features
        # TODO check for NCHW oder NHWC

        return 0

    def build_graph(self, *args):
        # TODO how to get inputs
        # TODO compute the loss functions

        return 0


