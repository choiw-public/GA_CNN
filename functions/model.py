from functions.module import Module
from functions.utils import get_shape
from numpy.random import randint
from numpy import prod
import tensorflow as tf


class Ancestor(Module):
    def __init__(self, train_data, common, name):
        self.common = common
        self.name = name
        self.conv_size = []
        self.conv_depth = []
        self.conv_stride = []
        self.model_length = randint(self.common.model_depth_range[0], self.common.model_depth_range[1] + 1)
        for i in range(self.model_length):
            self.conv_size.append(randint(self.common.conv_size_range[0], self.common.conv_size_range[1] + 1))
            self.conv_depth.append(randint(self.common.conv_depth_range[0], self.common.conv_depth_range[1] + 1))
            self.conv_stride.append(randint(self.common.conv_stride_range[0], self.common.conv_stride_range[1] + 1))
        logit = self.build(train_data["image"], True)

        # count number of parameters
        all_trainables = tf.trainable_variables()
        self.total_parameters = 0
        for variable in all_trainables:
            if "batch_normalization" not in variable.name:
                self.total_parameters += prod([int(para) for para in get_shape(variable)])

        self.loss = self.xntropy(logit, train_data["gt"])
        self.encode_gene()

    def build(self, image, is_train):
        net = image
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            for conv_size, conv_depth, conv_stride in zip(self.conv_size, self.conv_depth, self.conv_stride):
                net = self.conv_block(net, conv_size, conv_stride, conv_depth, is_train, False)
            _, h, w, c = get_shape(net)
            net = tf.layers.average_pooling2d(net, [h, w], 1)
            return tf.squeeze(self.conv_block(net, 1, 1, self.common.num_classes, is_train, True))

    def encode_gene(self):
        """
        convert int to bit
        """
        self.conv_size_bit = []
        self.conv_depth_bit = []
        self.conv_stride_bit = []
        self.model_bit = ""
        for size, depth, stride in zip(self.conv_size, self.conv_depth, self.conv_stride):
            conv_size_bit = format(size, "b").zfill(self.common.conv_size_bit_num)
            conv_depth_bit = format(depth, "b").zfill(self.common.conv_depth_bit_num)
            conv_stride_bit = format(stride, "b").zfill(self.common.conv_stride_bit_num)
            self.conv_size_bit.append(conv_size_bit)
            self.conv_depth_bit.append(conv_depth_bit)
            self.conv_stride_bit.append(conv_stride_bit)
            self.model_bit += conv_size_bit + conv_depth_bit + conv_stride_bit
        self.model_bit = self.model_bit.zfill(self.common.model_length_bit_num)


class Offspring(Ancestor, Module):
    def __init__(self, train_data, offspring_gene, common, name):
        self.name = name
        self.conv_size = offspring_gene["conv_size"]
        self.conv_depth = offspring_gene["conv_depth"]
        self.conv_stride = offspring_gene["conv_stride"]
        self.model_length = len(self.conv_size)
        super(Offspring, self).__init__(train_data, common, name)
        logit = self.build(train_data["image"], True)

        # count number of parameters
        all_trainables = tf.trainable_variables()
        self.total_parameters = 0
        for variable in all_trainables:
            if "batch_normalization" not in variable.name:
                self.total_parameters += prod([int(para) for para in get_shape(variable)])

        self.loss = self.xntropy(logit, train_data["gt"])
