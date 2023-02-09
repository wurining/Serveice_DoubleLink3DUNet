from lib.modality_pairing_net_mid import ModalityPairingNetTrain, AuxLossLayer, ModalityPairingLossLayer
from lib.modality_pairing_net import ModalityPairingNetTrain, AuxLossLayer, ModalityPairingLossLayer
from lib.simple_3d_unet import SimpleUnetModel
import numpy as np
import tensorflow as tf
import sys
sys.path.append("./")
"""
@Create by Rining Wu
@Email ml20r2w@leeds.ac.uk

"""


class Evaluate:
    def __init__(self) -> None:
        self.model_a_path = "model/group A.hdf5"
        self.model_b_path = "model/group B.hdf5"
        self.model_c_path = "model/group C.hdf5"

    def prepare(self):
        physical_gpus = tf.config.experimental.list_physical_devices("GPU")
        print(physical_gpus)
        tf.config.experimental.set_memory_growth(
            tf.config.experimental.list_physical_devices('GPU')[0], True)
        tf.config.run_functions_eagerly(True)

    def post_process(self, prediction):
        return np.argmax(prediction, axis=4)[0, :, :, :]

    def group_a(self, input_path):
        self.prepare()
        # load method 1:
        # test_image_batch, test_mask_batch = self.train_img_data_generator.__next__()
        # load method 2:
        # test_image_batch = np.load(f"{self.DATASET_DIR}/split_combined_data/train/images/image_BraTS2021_00030.npy")
        test_image_batch = np.load(input_path)
        test_mask_batch = np.zeros(test_image_batch.shape)
        input = [np.array([test_image_batch]), np.array([test_mask_batch])]

        model = tf.keras.models.load_model(self.model_a_path,
                                           compile=False,
                                           custom_objects={'ModalityPairingNetTrain': ModalityPairingNetTrain,
                                                           'AuxLossLayer': AuxLossLayer,
                                                           'ModalityPairingLossLayer': ModalityPairingLossLayer})
        return self.post_process(model.predict(input))

    def group_b(self, input_path):
        self.prepare()
        test_image_batch = np.load(input_path)
        test_mask_batch = np.zeros(test_image_batch.shape)
        input = [np.array([test_image_batch]), np.array([test_mask_batch])]

        model = tf.keras.models.load_model(self.model_b_path,
                                           compile=False,
                                           custom_objects={'ModalityPairingNetTrain': ModalityPairingNetTrain,
                                                           'AuxLossLayer': AuxLossLayer,
                                                           'ModalityPairingLossLayer': ModalityPairingLossLayer})
        return self.post_process(model.predict(input))

    def group_c(self, input_path):
        self.prepare()
        test_image_batch = np.load(input_path)
        input = [np.array([test_image_batch])]

        model = tf.keras.models.load_model(self.model_c_path,
                                           compile=False,
                                           custom_objects={'SimpleUnetModel': SimpleUnetModel})

        return self.post_process(model.predict(input))


if __name__ == '__main__':

    DATASET_DIR = "../tmp"
    train_img_dir = f"{DATASET_DIR}"

    evaluate = Evaluate()
    result_a = evaluate.group_a(f"{train_img_dir}/image_BraTS2021_00028.npy")
    # result_b = evaluate.group_b(f"{train_img_dir}/image_BraTS2021_00030.npy")
    # result_c = evaluate.group_c(f"{train_img_dir}/image_BraTS2021_00030.npy")
    print("finish")
