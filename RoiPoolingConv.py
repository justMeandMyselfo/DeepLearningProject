import tensorflow as tf
from keras.layers import Layer

class RoiPoolingConv(Layer):
    """ROI pooling layer for 2D inputs.

    Args:
        pool_size: int. Size of pooling region. pool_size = 7 gives 7x7 output.
        num_rois: Number of regions of interest.
        rois_mat: Array of shape (num_rois, 4) with RoI format (x, y, w, h)
    """

    def __init__(self, pool_size, num_rois, rois_mat, **kwargs):
        super(RoiPoolingConv, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.rois = tf.constant(rois_mat, dtype=tf.int32)

    def build(self, input_shape):
        self.nb_channels = input_shape[-1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_rois, self.pool_size, self.pool_size, self.nb_channels)

    def call(self, x):
        # x shape: (batch_size, height, width, channels)

        def process_single_roi(roi):
            x1, y1, w, h = tf.unstack(roi)
            x2 = x1 + w
            y2 = y1 + h

            region = x[:, y1:y2, x1:x2, :]  # (batch_size, roi_height, roi_width, channels)
            resized = tf.image.resize(region, (self.pool_size, self.pool_size))  # (batch_size, pool_size, pool_size, channels)
            return resized  # shape: (batch_size, pool_size, pool_size, channels)

        rois_pooled = tf.map_fn(
            process_single_roi,
            self.rois,
            fn_output_signature=tf.TensorSpec(shape=(None, self.pool_size, self.pool_size, self.nb_channels), dtype=tf.float32)
        )

        # rois_pooled: (num_rois, batch_size, pool_size, pool_size, channels)
        # -> transpose to (batch_size, num_rois, pool_size, pool_size, channels)
        rois_pooled = tf.transpose(rois_pooled, perm=[1, 0, 2, 3, 4])
        return rois_pooled

    def get_config(self):
        config = super(RoiPoolingConv, self).get_config()
        config.update({
            'pool_size': self.pool_size,
            'num_rois': self.num_rois,
            'rois_mat': self.rois.numpy().tolist()  # Serialize tensor as list
        })
        return config
