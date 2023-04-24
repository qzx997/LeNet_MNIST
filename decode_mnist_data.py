import numpy as np
import matplotlib.pyplot as plt
import struct
# import collect


def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'  # 因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  # 获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    fmt_image = '>' + str(
        image_size) + 'B'  # 图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image, offset, struct.calcsize(fmt_image))
    # print((num_images, num_rows, num_cols))
    images = np.empty((num_images, num_rows, num_cols))
    # plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        # print(images[i])
        offset += struct.calcsize(fmt_image)
    #        plt.imshow(images[i],'gray')
    #        plt.pause(0.00001)
    #        plt.show()
    # plt.show()

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


if __name__ == "__main__":
    train_image_file = "./raw/train-images.idx3-ubyte"
    train_label_file = "./raw/train-labels.idx1-ubyte"
    predict_image_file = "./raw/t10k-images.idx3-ubyte"
    predict_label_file = "./raw/t10k-labels.idx1-ubyte"

    train_images = decode_idx3_ubyte(train_image_file)
    train_labels = decode_idx1_ubyte(train_label_file)

    predict_images = decode_idx3_ubyte(predict_image_file)
    predict_labels = decode_idx1_ubyte(predict_label_file)

    label_train_str = ''
    for i in range(60000):
        if int(train_labels[i]) == 0 or int(train_labels[i]) == 1:
            label_train_str += "%d.png %d\n" % (i, int(train_labels[i]))
            plt.imshow(train_images[i], cmap='gray')
            plt.savefig("./train_image/%d.png" % i)
            # plt.show()
    print(label_train_str)
    f = open('train_label.txt', mode='w')
    f.write(label_train_str)
    f.close()
    print('train data generate done')

    label_predict_str = ''
    for i in range(10000):
        if int(predict_labels[i]) == 0 or int(predict_labels[i]) == 1:
            label_predict_str += "%d.png %d\n" % (i, int(predict_labels[i]))
            plt.imshow(predict_images[i], cmap='gray')
            plt.savefig("./predict_image/%d.png" % i)
            # plt.show()
    print(label_predict_str)
    f = open('predict_label.txt', mode='w')
    f.write(label_predict_str)
    f.close()
    print('predict data generate done')
