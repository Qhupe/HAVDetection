import tensorflow as tf
import keras.backend as K

from keras.layers import Input, Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers import LeakyReLU, BatchNormalization
from keras.models import Sequential, Model
from keras.regularizers import l2


def yolo_ConvBlock(input_tensor, num_filters, filter_size, strides=(1, 1)):
    padding = 'valid' if strides == (2, 2) else 'same'

    ### Layers
    x = Conv2D(num_filters, filter_size, strides, padding, use_bias=False, kernel_regularizer=l2(5e-4))(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def yolo_ResidualBlocks(input_tensor, num_filters, num_blocks):
    ### Layers
    x = ZeroPadding2D(((1, 0), (1, 0)))(input_tensor)  # left & top padding
    x = yolo_ConvBlock(x, num_filters, filter_size=(3, 3), strides=(2, 2))

    for _ in range(num_blocks):
        y = yolo_ConvBlock(x, num_filters // 2, filter_size=(1, 1), strides=(1, 1))
        y = yolo_ConvBlock(y, num_filters, filter_size=(3, 3), strides=(1, 1))
        x = Add()([x, y])

    return x


def yolo_OutputBlock(x, num_filters, out_filters):
    ### Layers
    x = yolo_ConvBlock(x, 1 * num_filters, filter_size=(1, 1), strides=(1, 1))
    x = yolo_ConvBlock(x, 2 * num_filters, filter_size=(3, 3), strides=(1, 1))
    x = yolo_ConvBlock(x, 1 * num_filters, filter_size=(1, 1), strides=(1, 1))
    x = yolo_ConvBlock(x, 2 * num_filters, filter_size=(3, 3), strides=(1, 1))
    x = yolo_ConvBlock(x, 1 * num_filters, filter_size=(1, 1), strides=(1, 1))

    y = yolo_ConvBlock(x, 2 * num_filters, filter_size=(3, 3), strides=(1, 1))
    y = Conv2D(filters=out_filters, kernel_size=(1, 1), strides=(1, 1),
               padding='same', use_bias=True, kernel_regularizer=l2(5e-4))(y)

    return x, y


def yolo_body(input_tensor, num_out_filters):
    x = yolo_ConvBlock(input_tensor, num_filters=32, filter_size=(3, 3), strides=(1, 1))

    x = yolo_ResidualBlocks(x, num_filters=64, num_blocks=1)
    x = yolo_ResidualBlocks(x, num_filters=128, num_blocks=2)
    x = yolo_ResidualBlocks(x, num_filters=256, num_blocks=8)
    x = yolo_ResidualBlocks(x, num_filters=512, num_blocks=8)
    x = yolo_ResidualBlocks(x, num_filters=1024, num_blocks=4)

    darknet = Model(input_tensor, x)

    x, y1 = yolo_OutputBlock(x, num_filters=512, out_filters=num_out_filters)

    x = yolo_ConvBlock(x, num_filters=256, filter_size=(1, 1), strides=(1, 1))
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = yolo_OutputBlock(x, num_filters=256, out_filters=num_out_filters)

    x = yolo_ConvBlock(x, num_filters=128, filter_size=(1, 1), strides=(1, 1))
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = yolo_OutputBlock(x, num_filters=128, out_filters=num_out_filters)

    model = Model(input_tensor, [y1, y2, y3])

    return model