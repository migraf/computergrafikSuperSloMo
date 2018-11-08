import tensorflow as tf
from tensorpack import ModelDesc


def conv_block(input, filter, kernel_size, stride, final_layer= False):
    out = tf.layers.conv2d(input, filters = filter, kernel_size=kernel_size, strides=stride)
    out = tf.layers.conv2d(out, filters=filter, kernel_size=kernel_size, strides=stride)
    out = tf.nn.leaky_relu(out, alpha=0.1)
    if final_layer:
        out = tf.layers.max_pooling2d(out, 2,2)
        return out
    return out

def upconv_block(input, filter, kernel_size, stride):
    # TODO is this correct
    out = tf.image.resize_bilinear(input, [input.shape[1], input.shape[2]])
    out = tf.layers.conv2d(out, filters=filter, kernel_size=kernel_size, strides=stride)
    out = tf.layers.conv2d(out, filters=filter, kernel_size=kernel_size, strides=stride)
    return out


# TODO Flow model and Flow refine model inside of a single model

# TODO maybe write a base u net model and change based on the input

class FlowModel(ModelDesc):
    # TODO what needs to be done at initalization
    def __init__(self, name):
        self.name = name

    def inputs(self):
        # TODO make placeholders for inputs
        return [tf.placeholder(tf.float32, (360,360), name="I0"),
                tf.placeholder(tf.float32, (360, 360), name="I1")]

    def hierarchy_layer_down(self, input, filter, kernel_size):
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

    def hierarchy_layer_up(self, input, filter):
        """
        Decoder Layer for U-Net architecture, doubles the size of the input
        and performs convolution
        :param input:
        :param filter:
        :return:
        """
        sizes = input.shape
        out = tf.image.resize_bilinear(input, [sizes[1]*2, sizes[2]*2])
        out = tf.layers.conv2d(out, filters=filter, kernel_size=3, strides=1)
        out = tf.nn.leaky_relu(out, alpha=0.1)
        out = tf.layers.conv2d(out, filters=filter, kernel_size=3, strides = 1)
        out = tf.nn.leaky_relu(out, alpha=0.1)

        return out

    def basic_flow(self, I0, I1):
        """
        Base Flow compution network, calculates the optical Flow F0_1 and F1_0
        between I0 and I1
        :param I0:
        :param I1:
        :return:
        """
        skip_connection = []
        X = tf.concat(I0, I1,axis=1)

        # U-Net Encoder

        # First Hierarchy Kernel Size 7
        out = self.hierarchy_layer_down(X, 32, 7)
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

        out = self.hierarchy_layer_up(out, 512)
        out = self.hierarchy_layer_up(out, 256)
        out = self.hierarchy_layer_up(out, 128)
        out = self.hierarchy_layer_up(out, 64)
        out = self.hierarchy_layer_up(out, 32)

        return out




