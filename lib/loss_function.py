import tensorflow as tf


class MainLossFunction(tf.keras.losses.Loss):
    def __init__(self, sample_weight=None, data_type=tf.float32):
        super().__init__(name='main_loss_function')
        self.sample_weight = sample_weight
        self.dtype_mix = data_type

    def __call__(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = self.sample_weight
        wt0, wt1, wt2, wt3 = sample_weight
        Norm_loss = DiceLoss(data_type=self.dtype_mix)(y_true[..., 0:1], y_pred[..., 0:1])
        NCR_loss = DiceLoss(data_type=self.dtype_mix)(y_true[..., 1:2], y_pred[..., 1:2])
        ED_loss = DiceLoss(data_type=self.dtype_mix)(y_true[..., 2:3], y_pred[..., 2:3])
        ET_loss = DiceLoss(data_type=self.dtype_mix)(y_true[..., 3:4], y_pred[..., 3:4])

        lambda_1 = 1 * (wt0 * Norm_loss + wt1 * NCR_loss + wt2 * ED_loss + wt3 * ET_loss) / (wt0 + wt1 + wt2 + wt3)
        lambda_2 = 1 * tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
        return lambda_1 + lambda_2


class AuxLossFunction(tf.keras.losses.Loss):
    def __init__(self, sample_weight=None, data_type=tf.float32):
        super().__init__(name='aux_loss_function')
        self.sample_weight = sample_weight
        self.dtype_mix = data_type

    def __call__(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            sample_weight = self.sample_weight
        wt0, wt1, wt2, wt3 = sample_weight
        Norm_loss = DiceLoss(data_type=self.dtype_mix)(y_true[..., 0:1], y_pred[..., 0:1])
        NCR_loss = DiceLoss(data_type=self.dtype_mix)(y_true[..., 1:2], y_pred[..., 1:2])
        ED_loss = DiceLoss(data_type=self.dtype_mix)(y_true[..., 2:3], y_pred[..., 2:3])
        ET_loss = DiceLoss(data_type=self.dtype_mix)(y_true[..., 3:4], y_pred[..., 3:4])

        lambda_1 = 1 * (wt0 * Norm_loss + wt1 * NCR_loss + wt2 * ED_loss + wt3 * ET_loss) / (wt0 + wt1 + wt2 + wt3)
        lambda_2 = 1 * tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
        return lambda_1 + lambda_2


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, data_type=tf.float32):
        super().__init__(name='dice_loss')
        self.dtype_mix = data_type

    def __call__(self, y_true, y_pred, sample_weight=None):
        result = tf.constant(0, dtype=self.dtype_mix)
        batch = y_true.shape[0]
        smooth = tf.constant(1e-5, dtype=tf.float32)
        scale = 1
        if scale != 1:
            if not batch:
                return result
            y_true = tf.multiply(y_true, scale)
            y_pred = tf.multiply(y_pred, scale)
            for i in range(batch):
                numerator = (2 * tf.reduce_sum(y_true[i] * y_pred[i])) / scale + smooth * scale
                denominator = tf.reduce_sum(y_true[i] + y_pred[i])
                result = tf.math.divide_no_nan(tf.cast(numerator, dtype=self.dtype_mix),
                                               tf.cast(denominator, dtype=self.dtype_mix)) + tf.cast(result,
                                                                                                     dtype=self.dtype_mix)
            return 1 - result / batch
        else:
            if not batch:
                return result
            for i in range(batch):
                numerator = (2.0 * tf.reduce_sum(y_true[i] * y_pred[i])) + smooth
                denominator = tf.reduce_sum(y_true[i] + y_pred[i])
                result = tf.math.divide_no_nan(tf.cast(numerator, dtype=self.dtype_mix),
                                               tf.cast(denominator, dtype=self.dtype_mix)) + tf.cast(result,
                                                                                                     dtype=self.dtype_mix)
            return 1 - result / batch


class ModalityPairingLoss(tf.keras.losses.Loss):
    def __init__(self, data_type=tf.float32):
        super().__init__(name='modality_pairing_loss')
        self.dtype_mix = data_type

    def __call__(self, y_true, y_pred, sample_weight=None):
        As, Bs = y_pred[0], y_pred[1]
        # Warning 如果使用float32进行计算，这里需要改变数据格式
        result = tf.constant(0, dtype=self.dtype_mix)
        batch = As.shape[0]
        if not batch:
            return result
        for i in range(batch):
            A, B = As[i], Bs[i]
            A_mean = tf.math.reduce_mean(A)
            B_mean = tf.math.reduce_mean(B)
            A_square_sum = tf.reduce_sum(tf.math.square(A - A_mean))
            B_square_sum = tf.reduce_sum(tf.math.square(B - B_mean))

            up = tf.reduce_sum(tf.multiply((A - A_mean), (B - B_mean)))
            down = tf.sqrt(tf.multiply(A_square_sum, B_square_sum))

            result = tf.add(result, tf.negative(tf.math.divide_no_nan(up, down)))
        return result / batch
