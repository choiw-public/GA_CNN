from tensorflow.layers import conv2d, batch_normalization
from tensorflow.nn import relu, softmax_cross_entropy_with_logits_v2
from tensorflow import cast, squeeze, int32, float32, one_hot, reshape, reduce_mean
from functions.utils import get_shape


class Module:
    @staticmethod
    def conv_block(tensor_in, kernel_size, stride, out_depth, is_train, use_bias):
        main_pipe = conv2d(tensor_in, out_depth, kernel_size, stride, "SAME", use_bias=use_bias)
        main_pipe = batch_normalization(main_pipe, training=is_train, fused=True)
        main_pipe = relu(main_pipe)
        return main_pipe

    def xntropy(self, logit, gt):
        if gt is None:
            raise ValueError('No label is given')
        gt = cast(squeeze(gt), int32)
        one_hot_gt = cast(one_hot(gt, self.common.num_classes, on_value=1.0, off_value=0.0), float32)

        if get_shape(logit) != get_shape(one_hot_gt):
            logit = squeeze(logit)
            if get_shape(logit) != get_shape(one_hot_gt):
                logit = reshape(logit, shape=[-1, self.common.num_classes])
                if get_shape(logit) != get_shape(one_hot_gt):
                    raise ValueError('unexpted logit shape')
        return reduce_mean(softmax_cross_entropy_with_logits_v2(labels=one_hot_gt, logits=logit))
