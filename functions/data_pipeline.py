import tensorflow as tf
from functions.utils import get_shape, list_getter


class DataPipeline:
    def __init__(self, src_dir, batch, is_train):
        self.tfrecord_feature = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                                 "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
        self._src_dir = src_dir
        self._batch = batch
        self.is_train = is_train
        self._build()

    def _preprocessing(self, image):
        # random resize 1~1.3
        if self.is_train:
            scale = tf.random.uniform([], minval=1.0, maxval=1.3, dtype=tf.float32)
            h, w, _ = get_shape(image)
            new_h = tf.cast(h, tf.float32) * scale
            new_w = tf.cast(w, tf.float32) * scale
            new_dim = tf.cast([new_h, new_w], tf.int32)
            image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0), new_dim, align_corners=True), [0])
            image = tf.image.random_crop(image, [28, 28, 1])
            # random rotate
            angle = tf.random.uniform((), minval=-20, maxval=20, dtype=tf.float32)
            image = tf.contrib.image.rotate(image, angle, interpolation='BILINEAR')
        else:
            image = tf.cast(image, tf.float32)
            image.set_shape([28, 28, 1])
        return image / 255.0

    def _parser(self, data):
        parsed = tf.parse_single_example(data, self.tfrecord_feature)
        image = tf.convert_to_tensor(tf.image.decode_jpeg(parsed["image"], channels=1))
        gt = tf.convert_to_tensor(parsed["label"])
        image = self._preprocessing(image)
        return {"image": image, "gt": gt}

    def _build(self):
        tfrecord_list = list_getter(self._src_dir, "tfrecord")
        if not tfrecord_list:
            raise ValueError("tfrecord is not given")
        data = tf.data.TFRecordDataset(tfrecord_list)
        data = data.shuffle(self._batch * 10)
        data = data.map(self._parser, 4).batch(self._batch, True)
        data = data.prefetch(2)
        iterator = data.make_initializable_iterator()
        data_batched = iterator.get_next()
        self.image = data_batched["image"]
        self.gt = data_batched["gt"]
        self.data_init = iterator.initializer
