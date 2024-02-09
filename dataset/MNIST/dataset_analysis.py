import struct
import numpy as np
import matplotlib.pyplot as plt

filename_train_images = './dataset/MNIST/train-images-idx3-ubyte'
filename_train_labels = './dataset/MNIST/train-labels-idx1-ubyte'
filename_test_images = './dataset/MNIST/t10k-images-idx3-ubyte'
filename_test_labels = './dataset/MNIST/t10k-labels-idx1-ubyte'


def show_pic(pic_array):
    plt.figure()
    plt.imshow(pic_array, 'gray')
    plt.show()


def load_train_images(filename=filename_train_images, debugger=0):
    # 读取二进制文件
    bin_file = open(filename, 'rb').read()

    # 解析头文件信息。魔数、图片数量、行、列
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_cols, num_rows = struct.unpack_from(fmt_header, bin_file, offset)  # return tuple
    if debugger:
        print('魔数：%d, 图片数量：%d, 行：%d, 列：%d' % (magic_number, num_images, num_cols, num_rows))

    # 解析数据集
    image_size = num_cols * num_rows
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    if debugger:
        print(fmt_image, offset, struct.calcsize(fmt_image))
    num_channels = 1
    images = np.empty((num_images, num_channels,num_rows, num_cols))

    for i in range(num_images):
        if (i + 1) % 10000 == 0 and debugger:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i][num_channels-1] = np.array(struct.unpack_from(fmt_image, bin_file, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)

    return images


def load_train_labels(filepath=filename_train_labels, debugger=0):
    # 读取二进制文件
    bin_file = open(filepath, 'rb').read()

    # 解析头文件信息。魔数、图片数量、行、列
    offset = 0
    fmt_header = '>ii'
    magic_number, num_labels = struct.unpack_from(fmt_header, bin_file, offset)  # return tuple
    if debugger:
        print('魔数：%d, 标签数量：%d' % (magic_number, num_labels))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_label = '>B'
    if debugger:
        print(fmt_label, offset, struct.calcsize(fmt_label))
    labels = np.empty(num_labels)

    for i in range(num_labels):
        if (i + 1) % 10000 == 0 and debugger:
            print('已解析 %d' % (i + 1) + '条标签')
            print(offset)
        labels[i] = np.array(struct.unpack_from(fmt_label, bin_file, offset))
        offset += struct.calcsize(fmt_label)

    return labels


def load_test_images(filename=filename_test_images, debugger=0):
    # 读取二进制文件
    bin_file = open(filename, 'rb').read()

    # 解析头文件信息。魔数、图片数量、行、列
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_cols, num_rows = struct.unpack_from(fmt_header, bin_file, offset)  # return tuple
    if debugger:
        print('魔数：%d, 图片数量：%d, 行：%d, 列：%d' % (magic_number, num_images, num_cols, num_rows))

    # 解析数据集
    image_size = num_cols * num_rows
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    if debugger:
        print(fmt_image, offset, struct.calcsize(fmt_image))
    num_channels = 1
    images = np.empty((num_images, num_channels, num_rows, num_cols))

    for i in range(num_images):
        if (i + 1) % 2000 == 0 and debugger:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i][num_channels-1] = np.array(struct.unpack_from(fmt_image, bin_file, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)

    return images


def load_test_labels(filepath=filename_test_labels, debugger=0):
    # 读取二进制文件
    bin_file = open(filepath, 'rb').read()

    # 解析头文件信息。魔数、图片数量、行、列
    offset = 0
    fmt_header = '>ii'
    magic_number, num_labels = struct.unpack_from(fmt_header, bin_file, offset)  # return tuple
    if debugger:
        print('魔数：%d, 标签数量：%d' % (magic_number, num_labels))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_label = '>B'
    if debugger:
        print(fmt_label, offset, struct.calcsize(fmt_label))
    labels = np.empty(num_labels)

    for i in range(num_labels):
        if (i + 1) % 2000 == 0 and debugger:
            print('已解析 %d' % (i + 1) + '条标签')
            print(offset)
        labels[i] = np.array(struct.unpack_from(fmt_label, bin_file, offset))
        offset += struct.calcsize(fmt_label)

    return labels
