# define a simple data batch
from collections import namedtuple
from functools import lru_cache
from os.path import join
from time import time

import cv2
import mxnet as mx
import numpy as np
import pandas as pd
import requests

from logger import logger
from retrieval.retrieval.retrieval import Retrieval_model
from ssd import demo

Batch = namedtuple('Batch', ['data'])


class __model__():
    def __init__(self):
        self.table = pd.read_excel('data.xlsx')
        self.DEBUG = False

    @lru_cache(maxsize=5)
    def get_mod(self, folder_name, ctx, checkpoint_name=None, batch_size=None, longth_=None, width_=None):
        if folder_name not in ['ssd', 'retrieval']:
            logger.info('load model')
            sym, arg_params, aux_params = mx.model.load_checkpoint(join(folder_name, checkpoint_name), 0)
            mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
            mod.bind(for_training=False, data_shapes=[('data', (batch_size, 3, longth_, width_))],
                     label_shapes=mod._label_shapes)
            mod.set_params(arg_params, aux_params, allow_missing=True)
        elif folder_name == 'ssd':
            logger.info('load ssd model')
            mod = demo.get_ssd_model(ctx=ctx)
        elif folder_name == 'retrieval':
            logger.info('load retrieval model')
            mod = Retrieval_model(ctx=ctx)
        else:
            raise NotImplementedError('No Such model')
        return mod

    @lru_cache(maxsize=16)
    def get_img(self, url, model_level, img_path=None):
        if img_path is None:
            r = requests.get(url)
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(img_path)

        ratio = 1280 / img.shape[0]
        origin_img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            raise Exception("download fail,retry")
        mark = False
        if img.shape[0] < img.shape[1]:
            img = cv2.transpose(img)
            mark = True
        if model_level in [1, 10001]:
            longer_side_length = 160
        elif model_level in [2, 10002, 10003]:
            longer_side_length = 320
        if model_level in [10002,10003]:
            longer_side_length = 640
            if mark:
                img = cv2.transpose(img)
        ratio = longer_side_length / img.shape[0]
        img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
        # convert into format (batch, RGB, width, height)
        return img, origin_img

    @lru_cache(maxsize=16)
    def ssd(self, image_url, ctx):
        folder_name = 'ssd'
        tic = time()
        mod = self.get_mod(folder_name=folder_name, ctx=ctx)
        logger.info('use {}s to get model '.format(str(time() - tic)))
        img, _ = self.get_img(image_url, 10002)
        return mod.detect_and_return(img, thresh=0.5)

    @lru_cache(maxsize=16)
    def retrieval(self, image_url, color_level=0, style_level=0,ctx=mx.cpu(),database='skirt'):
        tic = time()
        _det, img = self.ssd(image_url, ctx=ctx)
        tic2 = time()
        logger.info('use {}s to ssd '.format(str(tic2 - tic)))
        #img,_ = self.get_img(image_url,model_level=10003)
        if img is None:
            logger.info('no skirt detected')
            return ''
        folder_name = 'retrieval'
        mod = self.get_mod(folder_name=folder_name, ctx=ctx)
        res = mod.search_database(img,color_level,style_level,database)
        logger.info('use {}s to search database '.format(str(time() - tic2)))
        return res

    # @lru_cache()
    def do_multi_predict(self, image_url, model_level, img_id, first_class_id=None, num_id=None, batch_size=1, tt=1,
                         ctx=mx.cpu(), img_show=False):
        name_dict = {1: 'first', 2: join('second', '{}'.format(str(first_class_id))), 10001: 'detect'}
        folder_name = name_dict[model_level]
        resnet_version = '152'

        if model_level == 2:
            mu = 2
            width = 110 * mu
            longth = int(160 / 8 * 7 * mu)
            longth_ = int(longth / 8 * 6)
            width_ = int(width / 8 * 6)
            augs = [mx.image.CenterCropAug(size=(width, longth)), mx.image.ForceResizeAug((width_, longth_))]
        elif model_level == 1:
            longth_ = int(160 / 8 * 7)
            width_ = int(110 / 8 * 7)
            augs = []
        elif model_level == 10001:
            longth_ = int(160 / 8 * 7)
            width_ = int(110)
            augs = []
        else:
            raise Exception('No Such Model')

        checkpoint_name = 'resnet{}'.format(resnet_version)

        augs += mx.image.CreateAugmenter(data_shape=(3, longth_, width_), rand_crop=False, rand_resize=False,
                                         rand_mirror=True, brightness=0.125, contrast=0.125, rand_gray=0.05,
                                         saturation=0.125, pca_noise=0.125, inter_method=10)
        t0 = time()
        final_res = None

        mod = self.get_mod(folder_name=folder_name, checkpoint_name=checkpoint_name, ctx=ctx, batch_size=batch_size,
                           longth_=longth_, width_=width_)
        t1 = (time())
        logger.info('use {}s to get model '.format(str(t1 - t0)))

        img, origin_img = self.get_img(url=image_url, model_level=model_level)
        for i in range(tt):
            img_nd = mx.nd.array(img)
            for aug in augs:
                img_nd = aug(img_nd)
            img_after_parse = cv2.cvtColor(img_nd.asnumpy(), cv2.COLOR_RGB2BGR)
            img_nd = mx.nd.swapaxes(img_nd, 0, 2)
            img_nd = mx.nd.swapaxes(img_nd, 1, 2)
            img_nd = mx.nd.array(img_nd.asnumpy()[np.newaxis, :])
            val_iter = mx.io.NDArrayIter([img_nd], None, batch_size=batch_size)
            res = mod.predict(val_iter, always_output_list=True)
            if final_res is None:
                final_res = res
            else:
                final_res += res
        logger.info('use {}s to predict'.format(str(time() - t1)))

        res_temp = mx.nd.zeros(final_res[0].shape, ctx=ctx)
        for i in final_res:
            res_temp += i
        res_temp /= len(final_res)
        if model_level == 2:
            t = self.table[self.table.cid == int(first_class_id)]
            t.bid = t.bid.apply(str)
            t = t.sort_values(by='bid')
            if self.DEBUG:
                return_list = t.bname.tolist()
            else:
                return_list = t.bid.tolist()
        elif model_level == 1:
            t = self.table[self.table.cid != 2]
            t = t[t.cid != 9]
            t = t.groupby('cid').head(1)
            t.cid = t.cid.apply(str)
            t = t.sort_values(by='cid')
            if self.DEBUG:
                return_list = t.cname.tolist()
            else:
                return_list = t.cid.tolist()
        elif model_level == 10001:
            pass
        else:
            raise Exception("no such model level!")

        res_numpy = res_temp[0].asnumpy()
        logger.info("res {}".format(str(res_numpy)))
        if model_level == 10001:
            return int(res_numpy.argmax())

        if img_show:
            cv2.imwrite('1.jpg', origin_img)
            cv2.imwrite('1pm.jpg', img)
            cv2.imwrite('1p.jpg', img_after_parse)
        if max(res_numpy) > 0.80:
            return [return_list[res_numpy.argmax()]]
        temp = []
        if str(model_level) == '1':
            return_longth = 3
        if str(model_level) == '2':
            return_longth = 5
        for i in res_numpy.argsort()[::-1][0:return_longth]:
            temp.append(return_list[i])
        return temp


model = __model__()
if __name__ == '__main__':
    model = __model__()
    print(type(int(model.do_multi_predict(
        'https://static.leiphone.com/uploads/new/article/740_740/201711/5a002bed9cf7f.png?imageMogr2/format/jpg/quality/90',
        1, tt=5))))
