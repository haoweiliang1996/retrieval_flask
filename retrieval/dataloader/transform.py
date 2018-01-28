import mxnet as mx
import numpy as np
from mxnet import image
from mxnet import nd

augs = image.CreateAugmenter(data_shape=(3, 224, 224), rand_mirror=True, rand_crop=True, rand_resize=False,
                             brightness=0.125, contrast=0.125, saturation=0.125)
augs_val = image.CreateAugmenter(data_shape=(3, 224, 224), rand_crop=False, rand_mirror=False)
cropaug = mx.image.RandomSizedCropAug(size=(224, 224), min_area=0.5, ratio=[0.75, 1.33333])
resize_short_aug = mx.image.ResizeAug(size=224)
augs.insert(0, resize_short_aug)
augs_val.insert(0, resize_short_aug)


def transform(data, label):
    for aug in augs:
        data = aug(data)
    return nd.transpose(mx.image.color_normalize(data.astype(np.float32) / 255,
                                                 mean=mx.nd.array([0.485, 0.456, 0.406]),
                                                 std=mx.nd.array([0.229, 0.224, 0.225])), (2, 0, 1)), label


def transform_val(data, label):
    for aug in augs_val:
        data = aug(data)
    return nd.transpose(mx.image.color_normalize(data.astype(np.float32) / 255,
                                                 mean=mx.nd.array([0.485, 0.456, 0.406]),
                                                 std=mx.nd.array([0.229, 0.224, 0.225])), (2, 0, 1)), label


def transform_test(img):
    def fix_crop(src, tu):
        x, y = tu
        return mx.image.fixed_crop(src, x, y, 224, 224)

    img = mx.image.resize_short(img, 224)
    height, width, _ = img.shape
    imgs = [fix_crop(img, tu) for tu in [(0, 0), (width - 224, 0), (0, height - 224), (width - 224, height - 224)]]
    imgs.append(mx.image.center_crop(img, (224, 224))[0])
    temp = []
    for i in imgs:
        temp.append(nd.transpose(mx.image.color_normalize(i.astype(np.float32) / 255,
                                                     mean=mx.nd.array([0.485, 0.456, 0.406]),
                                                     std=mx.nd.array([0.229, 0.224, 0.225])), (2, 0, 1)))
    return nd.stack(*temp)


def transform_histgram(data, label):
    # return data,label
    data = mx.image.CenterCropAug((112, 112))(data)
    return data, label


if __name__ == '__main__':
    for i in augs:
        print(i.dumps())
