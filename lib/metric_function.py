import tensorflow as tf
from tensorflow import keras
from lib.loss_function import DiceLoss


# Dice分数---Hausdorff 95分数
# 需要得到每个类分别的和总体的
# 都是由mask和pred这一个输出得出

class DiceScore(keras.metrics.Metric):
    def __init__(self, name="dice_score", sample_weight=None, data_type=tf.double, **kwargs):
        super(DiceScore, self).__init__(name=name, **kwargs)
        self.dtype_mix = data_type
        self.total_score = self.add_weight(name="total_score", initializer="zeros", dtype=data_type)
        self.NCR = self.add_weight(name="NCR_score", initializer="zeros", dtype=data_type)
        self.ED = self.add_weight(name="ED_score", initializer="zeros", dtype=data_type)
        self.ET = self.add_weight(name="ET_score", initializer="zeros", dtype=data_type)
        #  0     1     2     3
        #  Normal (Norm — label 0)
        #  the necrotic tumor core (NCR — label 1)
        #  the peritumoral edematous/invaded tissue (ED — label 2)
        #  the GD-enhancing tumor (ET — label 4=3)

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        Norm_loss = DiceLoss(data_type=self.dtype_mix)(y_true[..., 0:1], y_pred[..., 0:1])
        NCR_loss = DiceLoss(data_type=self.dtype_mix)(y_true[..., 1:2], y_pred[..., 1:2])
        ED_loss = DiceLoss(data_type=self.dtype_mix)(y_true[..., 2:3], y_pred[..., 2:3])
        ET_loss = DiceLoss(data_type=self.dtype_mix)(y_true[..., 3:4], y_pred[..., 3:4])
        total_score_loss = Norm_loss + NCR_loss + ED_loss + ET_loss

        self.total_score.assign((4 - total_score_loss) / 4)
        self.NCR.assign(1 - NCR_loss)
        self.ED.assign(1 - ED_loss)
        self.ET.assign(1 - ET_loss)

    def result(self):
        return self.total_score, self.NCR, self.ED, self.ET

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total_score.assign(0.0)
        self.NCR.assign(0.0)
        self.ED.assign(0.0)
        self.ET.assign(0.0)


class Sensitivity(keras.metrics.Metric):
    def __init__(self, name="sensitivity", sample_weight=None, data_type=tf.double, **kwargs):
        super(Sensitivity, self).__init__(name=name, **kwargs)
        self.dtype_mix = data_type
        self.total_score = self.add_weight(name="total_score", initializer="zeros", dtype=data_type)
        self.NCR = self.add_weight(name="NCR_score", initializer="zeros", dtype=data_type)
        self.ED = self.add_weight(name="ED_score", initializer="zeros", dtype=data_type)
        self.ET = self.add_weight(name="ET_score", initializer="zeros", dtype=data_type)

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        Norm_loss = SensitivityLoss(data_type=self.dtype_mix)(y_true[..., 0:1], y_pred[..., 0:1])
        NCR_loss = SensitivityLoss(data_type=self.dtype_mix)(y_true[..., 1:2], y_pred[..., 1:2])
        ED_loss = SensitivityLoss(data_type=self.dtype_mix)(y_true[..., 2:3], y_pred[..., 2:3])
        ET_loss = SensitivityLoss(data_type=self.dtype_mix)(y_true[..., 3:4], y_pred[..., 3:4])
        total_score_loss = Norm_loss + NCR_loss + ED_loss + ET_loss

        self.total_score.assign(total_score_loss / 4)
        self.NCR.assign(NCR_loss)
        self.ED.assign(ED_loss)
        self.ET.assign(ET_loss)

    def result(self):
        return self.total_score, self.NCR, self.ED, self.ET

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total_score.assign(0.0)
        self.NCR.assign(0.0)
        self.ED.assign(0.0)
        self.ET.assign(0.0)


class SensitivityLoss(tf.keras.losses.Loss):
    def __init__(self, data_type=tf.float32):
        super().__init__(name='sensitivity_loss')
        self.dtype_mix = data_type

    def __call__(self, y_true, y_pred, sample_weight=None):
        result = tf.constant(0, dtype=self.dtype_mix)
        batch = y_true.shape[0]
        smooth = 1e-5
        if not batch:
            return result
        for i in range(batch):
            numerator = tf.reduce_sum(y_true[i] * y_pred[i]) + smooth
            denominator = tf.reduce_sum(y_true[i])
            result = tf.math.divide_no_nan(tf.cast(numerator, dtype=self.dtype_mix),
                                           tf.cast(denominator, dtype=self.dtype_mix)) + tf.cast(result,
                                                                                                 dtype=self.dtype_mix)
            return result / batch


class Specificity(keras.metrics.Metric):
    def __init__(self, name="specificity", sample_weight=None, data_type=tf.double, **kwargs):
        super(Specificity, self).__init__(name=name, **kwargs)
        self.dtype_mix = data_type
        self.total_score = self.add_weight(name="total_score", initializer="zeros", dtype=data_type)
        self.NCR = self.add_weight(name="NCR_score", initializer="zeros", dtype=data_type)
        self.ED = self.add_weight(name="ED_score", initializer="zeros", dtype=data_type)
        self.ET = self.add_weight(name="ET_score", initializer="zeros", dtype=data_type)

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        Norm_loss = SpecificityLoss(data_type=self.dtype_mix)(y_true[..., 0:1], y_pred[..., 0:1])
        NCR_loss = SpecificityLoss(data_type=self.dtype_mix)(y_true[..., 1:2], y_pred[..., 1:2])
        ED_loss = SpecificityLoss(data_type=self.dtype_mix)(y_true[..., 2:3], y_pred[..., 2:3])
        ET_loss = SpecificityLoss(data_type=self.dtype_mix)(y_true[..., 3:4], y_pred[..., 3:4])
        total_score_loss = Norm_loss + NCR_loss + ED_loss + ET_loss

        self.total_score.assign(total_score_loss / 4)
        self.NCR.assign(NCR_loss)
        self.ED.assign(ED_loss)
        self.ET.assign(ET_loss)

    def result(self):
        return self.total_score, self.NCR, self.ED, self.ET

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total_score.assign(0.0)
        self.NCR.assign(0.0)
        self.ED.assign(0.0)
        self.ET.assign(0.0)


class SpecificityLoss(tf.keras.losses.Loss):
    def __init__(self, data_type=tf.float32):
        super().__init__(name='specificity_loss')
        self.dtype_mix = data_type

    def __call__(self, y_true, y_pred, sample_weight=None):
        result = tf.constant(0, dtype=self.dtype_mix)
        batch = y_true.shape[0]
        smooth = 1e-5
        if not batch:
            return result
        for i in range(batch):
            numerator = tf.reduce_sum((1 - y_true[i]) * y_pred[i]) + smooth
            denominator = tf.reduce_sum((1 - y_true[i]))
            result = 1 - tf.math.divide_no_nan(tf.cast(numerator, dtype=self.dtype_mix),
                                           tf.cast(denominator, dtype=self.dtype_mix)) + tf.cast(result,
                                                                                                 dtype=self.dtype_mix)
        return result / batch


# Dice分数---Hausdorff 95分数
# 需要得到每个类分别的和总体的
# 都是由mask和pred这一个输出得出
def convert_to_tuple_list(l):
    # n * N-d
    if l.shape[0] is None:
        return []
    return [tuple(i) for i in l]