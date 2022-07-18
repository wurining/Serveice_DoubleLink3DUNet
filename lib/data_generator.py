import numpy as np

import random


# 加载图片 添加至np数组
def load_img(img_dir_path, img_name_list):
    images = []

    for i, img_name in enumerate(img_name_list):
        if img_name.split('.')[1] == 'npy':
            image = np.load(img_dir_path + img_name)
            images.append(image)

    images = np.array(images)
    return images


# 加载Batch
def image_loader(img_dir, img_list, mask_dir, mask_list, batch_size, shuffle=False):
    list_length = len(img_list)
    list_index = list(range(list_length))

    # keras 要求的Generator需要可以无限迭代
    while True:
        batch_start = 0
        batch_end = batch_size
        if shuffle:
            random.shuffle(list_index)
        # 取单个Batch
        while batch_start < list_length:
            limit = min(batch_end, list_length)
            # list_index[batch_start:limit]
            X = load_img(img_dir, [img_list[j] for j in list_index[batch_start:limit]])
            Y = load_img(mask_dir, [mask_list[k] for k in list_index[batch_start:limit]])
            yield X, Y  # a tuple with two numpy arrays with batch_size samples
            # 下一组Batch的Cursor
            batch_start += batch_size
            batch_end += batch_size
