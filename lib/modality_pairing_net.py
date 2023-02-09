"""
@author: Rining Wu
"""
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv3D, concatenate, Conv3DTranspose, LeakyReLU, Softmax, Dropout, MaxPooling3D
from lib.loss_function import AuxLossFunction, ModalityPairingLoss



class AuxLossLayer(tf.keras.layers.Layer):
    def call(self, inputs, num_classes=4, data_type=tf.float32):
        # loss_weight = 0.625
        loss_weight = 0.25
        wt0, wt1, wt2, wt3 = (0.26, 33.8, 33.8, 24.87)
        aux_a_output = inputs[0]
        aux_b_output = inputs[1]
        mask = inputs[2]
        times = {
            1: 0,  # 128
            2: 1,  # 64
            4: 2,  # 32
            8: 3,  # 16
            16: 4,  # 8
            32: 5,  # 4
        }
        # 1把input做卷积
        o = output_conv_block(concatenate([aux_b_output, aux_a_output]), num_classes)
        # 1把Mask做卷积
        times_mask_dic_a = int(mask.shape[1] / aux_a_output.shape[1])
        for i in range(times.get(times_mask_dic_a)):
            mask = MaxPooling3D(pool_size=(2, 2, 2))(mask)
        # 2跟旁路输出做Aux
        loss = AuxLossFunction(sample_weight=[wt0, wt1, wt2, wt3], data_type=data_type)(mask, o)
        self.add_loss(loss * loss_weight)
        return aux_a_output, aux_b_output, mask


class ModalityPairingLossLayer(tf.keras.layers.Layer):
    def call(self, inputs, data_type=tf.float32):
        loss_weight = 0.5
        wt0, wt1, wt2, wt3 = (0.26, 33.8, 33.8, 24.87)
        aux_a_output = inputs[0]
        aux_b_output = inputs[1]
        outputs = inputs[2]
        loss = ModalityPairingLoss(data_type=data_type)([0], [aux_a_output, aux_b_output])
        # print('ModalityPairingLossLayer', loss)
        self.add_loss(loss * loss_weight)
        return outputs


def conv_block(i, output_channels, kernel_size=(3, 3, 3), dropout=True, dropout_rate=0.1, name=None):
    kernel_initializer = 'he_uniform'
    o = Conv3D(output_channels / 2,
               kernel_size=kernel_size,
               kernel_initializer=kernel_initializer,
               activation=LeakyReLU(name=name + '_LeakyReLU1', alpha=0.1),
               padding='same',
               name=name + '_Conv1')(i)
    o = tfa.layers.InstanceNormalization(name=name + '_InstanceNorm1')(o)
    if dropout:
        o = Dropout(dropout_rate, name=f'{name}_Dropout')(o)
    o = Conv3D(output_channels,
               kernel_size=kernel_size,
               kernel_initializer=kernel_initializer,
               activation=LeakyReLU(name=name + '_LeakyReLU2', alpha=0.1),
               padding='same',
               name=name + '_Conv2')(o)
    o = tfa.layers.InstanceNormalization(name=name + '_InstanceNorm2')(o)
    return o


def output_conv_block(i, output_channels, kernel_size=(1, 1, 1), name=None):
    o = Conv3D(output_channels, kernel_size, activation='softmax', name=name)(i)
    return o


def get_main_net(IMG_HEIGHT=128, IMG_WIDTH=128, IMG_DEPTH=128, IMG_CHANNELS=4, num_classes=4, data_type=tf.float32):
    is_aux_loss = True
    is_dropout = True
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS), name='inputs')
    mask = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS), name='mask')
    #      Path A: T2(2) and Flair(3)
    #      Path B: T1(0) and T1ce(1)
    # Path-A input
    path_a = inputs[::, ::, ::, ::, 2:4:1]
    # Path-B input
    path_b = inputs[::, ::, ::, ::, 0:2:1]
    layer_1_output_channel = 32  # 16  # 8 #32
    layer_2_output_channel = 64  # 32  # 16 #64
    layer_3_output_channel = 128  # 64  # 32 #128
    layer_4_output_channel = 196  # 128  # 64 #256
    layer_5_output_channel = 256  # 256  # 128 #320

    # Encoder
    c1a_s32 = conv_block(path_a,
                         output_channels=layer_1_output_channel,
                         dropout=is_dropout,
                         dropout_rate=0.1,
                         name='L1DownA')
    c1b_s32 = conv_block(path_b,
                         output_channels=layer_1_output_channel,
                         dropout=is_dropout,
                         dropout_rate=0.1,
                         name='L1DownB')
    feature_a = MaxPooling3D(pool_size=(2, 2, 2))(c1a_s32)
    feature_b = MaxPooling3D(pool_size=(2, 2, 2))(c1b_s32)

    c2a_s64 = conv_block(concatenate([feature_a, feature_b]),
                         output_channels=layer_2_output_channel,
                         dropout=is_dropout,
                         dropout_rate=0.1,
                         name='L2DownA')
    c2b_s64 = conv_block(concatenate([feature_b, feature_a]),
                         output_channels=layer_2_output_channel,
                         dropout=is_dropout,
                         dropout_rate=0.1,
                         name='L2DownB')
    feature_a = MaxPooling3D(pool_size=(2, 2, 2))(c2a_s64)
    feature_b = MaxPooling3D(pool_size=(2, 2, 2))(c2b_s64)

    c3a_s128 = conv_block(concatenate([feature_a, feature_b]),
                          output_channels=layer_3_output_channel,
                          dropout=is_dropout,
                          dropout_rate=0.1,
                          name='L3DownA')
    c3b_s128 = conv_block(concatenate([feature_b, feature_a]),
                          output_channels=layer_3_output_channel,
                          dropout=is_dropout,
                          dropout_rate=0.1,
                          name='L3DownB')
    feature_a = MaxPooling3D(pool_size=(2, 2, 2))(c3a_s128)
    feature_b = MaxPooling3D(pool_size=(2, 2, 2))(c3b_s128)

    c4a_s256 = conv_block(concatenate([feature_a, feature_b]),
                          output_channels=layer_4_output_channel,
                          dropout=is_dropout,
                          dropout_rate=0.2,
                          name='L4DownA')
    c4b_s256 = conv_block(concatenate([feature_b, feature_a]),
                          output_channels=layer_4_output_channel,
                          dropout=is_dropout,
                          dropout_rate=0.2,
                          name='L4DownB')
    feature_a = MaxPooling3D(pool_size=(2, 2, 2))(c4a_s256)
    feature_b = MaxPooling3D(pool_size=(2, 2, 2))(c4b_s256)

    # Bridge
    c5a_s320 = conv_block(concatenate([feature_a, feature_b]),
                          output_channels=layer_5_output_channel,
                          dropout=is_dropout,
                          dropout_rate=0.2,
                          name='L5DownA')
    c5b_s320 = conv_block(concatenate([feature_b, feature_a]),
                          output_channels=layer_5_output_channel,
                          dropout=is_dropout,
                          dropout_rate=0.2,
                          name='L5DownB')

    # Decoder
    feature_a = Conv3DTranspose(layer_4_output_channel, (3, 3, 3), strides=(2, 2, 2), padding='same')(c5a_s320)
    feature_b = Conv3DTranspose(layer_4_output_channel, (3, 3, 3), strides=(2, 2, 2), padding='same')(c5b_s320)
    c6a = conv_block(concatenate([feature_a, c4a_s256, c4b_s256, feature_b]),
                     output_channels=layer_4_output_channel,
                     dropout=is_dropout,
                     dropout_rate=0.2,
                     name='L4UpA')
    c6b = conv_block(concatenate([feature_b, c4b_s256, c4a_s256, feature_a]),
                     output_channels=layer_4_output_channel,
                     dropout=is_dropout,
                     dropout_rate=0.2,
                     name='L4UpB')
    # auxiliary outputs
    if is_aux_loss:
        c6a, c6b, no_meaning = AuxLossLayer()([c6a, c6b, mask], data_type=data_type)

    feature_a = Conv3DTranspose(layer_3_output_channel, (3, 3, 3), strides=(2, 2, 2), padding='same')(c6a)
    feature_b = Conv3DTranspose(layer_3_output_channel, (3, 3, 3), strides=(2, 2, 2), padding='same')(c6b)
    del c6a
    del c6b
    c7a = conv_block(concatenate([feature_a, c3a_s128, c3b_s128, feature_b]),
                     output_channels=layer_3_output_channel,
                     dropout=is_dropout,
                     dropout_rate=0.1,
                     name='L3UpA')
    c7b = conv_block(concatenate([feature_b, c3b_s128, c3a_s128, feature_a]),
                     output_channels=layer_3_output_channel,
                     dropout=is_dropout,
                     dropout_rate=0.1,
                     name='L3UpB')
    # auxiliary outputs
    if is_aux_loss:
        c7a, c7b, no_meaning = AuxLossLayer()([c7a, c7b, mask], data_type=data_type)

    feature_a = Conv3DTranspose(layer_2_output_channel, (3, 3, 3), strides=(2, 2, 2), padding='same')(c7a)
    feature_b = Conv3DTranspose(layer_2_output_channel, (3, 3, 3), strides=(2, 2, 2), padding='same')(c7b)
    del c7a
    del c7b
    c8a = conv_block(concatenate([feature_a, c2a_s64, c2b_s64, feature_b]),
                     output_channels=layer_2_output_channel,
                     dropout=is_dropout,
                     dropout_rate=0.1,
                     name='L2UpA')
    c8b = conv_block(concatenate([feature_b, c2b_s64, c2a_s64, feature_a]),
                     output_channels=layer_2_output_channel,
                     dropout=is_dropout,
                     dropout_rate=0.1,
                     name='L2UpB')
    # auxiliary outputs
    if is_aux_loss:
        c8a, c8b, no_meaning = AuxLossLayer()([c8a, c8b, mask], data_type=data_type)

    feature_a = Conv3DTranspose(layer_1_output_channel, (3, 3, 3), strides=(2, 2, 2), padding='same')(c8a)
    feature_b = Conv3DTranspose(layer_1_output_channel, (3, 3, 3), strides=(2, 2, 2), padding='same')(c8b)
    del c8a
    del c8b
    c9a = conv_block(concatenate([feature_a, c1a_s32, c1b_s32, feature_b]),
                     output_channels=layer_1_output_channel,
                     dropout=is_dropout,
                     dropout_rate=0.1,
                     name='L1UpA')
    c9b = conv_block(concatenate([feature_b, c1b_s32, c1a_s32, feature_a]),
                     output_channels=layer_1_output_channel,
                     dropout=is_dropout,
                     dropout_rate=0.1,
                     name='L1UpB')
    # auxiliary outputs
    # if is_aux_loss:
    #     c9a, c9b, no_meaning = AuxLossLayer()([c9a, c9b, mask], data_type=data_type)

    # fusion of the result
    # A_f = output_conv_block(c9a, num_classes, name='a')
    # B_f = output_conv_block(c9b, num_classes, name='b')
    outputs = output_conv_block(concatenate([c9b, c9a]), num_classes, name='classes')

    outputs = ModalityPairingLossLayer()([Softmax(trainable=False)(c9a),
                                          Softmax(trainable=False)(c9b),
                                          outputs],
                                         data_type=data_type)
    del c9a
    del c9b

    # outputs_list = [outputs, A_f, B_f]
    outputs_list = [outputs]
    final_model = ModalityPairingNetTrain(inputs=[inputs, mask], outputs=outputs_list)
    return final_model


class ModalityPairingNetTrain(tf.keras.Model):
    def train_step(self, data):
        # shape(batch,W,H,D,Channel)
        x, y = data  # data的结构取决于模型跟传给fit的数据结构

        with tf.GradientTape() as tape:
            y_pred = self([x, y], training=True)  # 前向计算
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        # 计算梯度
        grads = tape.gradient(loss, trainable_vars)
        # 更新参数
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        # 更新度量
        self.compiled_metrics.update_state(y, y_pred)
        # 返回度量结果，度量中包括loss
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self([x, y], training=False)

        # Unpack data
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


if __name__ == "__main__":
    # Test if everything is working ok.

    model_instance = get_main_net(128, 128, 128, 4, 4)
    # model_instance = get_main_net(64, 64, 64, 4, 4)
    model_instance.compile()
    model_instance.summary()
    # X = np.random.random(1 * 128 * 128 * 128 * 4).reshape(1, 128, 128, 128, 4)
    # Y = np.random.random(1 * 128 * 128 * 128 * 4).reshape(1, 128, 128, 128, 4)
    # model_instance.fit(x=X, y=Y, batch_size=1, epochs=1)
    # from keras.utils.vis_utils import plot_model

    # plot_model(model_instance, "output/get_main_net_final_and_dropout_maxpool.png", dpi=96, show_shapes=False)
    # print(model_instance.input_shape)
    # print(model_instance.output_shape)
    # for i in model_instance.layers:
    #     print(f">input:{i.input_shape}\t{i.name}\t>output:{i.output_shape}")
