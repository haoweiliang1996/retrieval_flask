'''
    允许自定义输入的文件库，支持保存本地索引库，argmax
    do PCA,and l2_norm
'''

import os
import sys
import pickle
from logger import logger

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '../../'))
sys.path.append(os.path.join(curr_path, '..'))
import os.path as osp

# define a simple data batch

json_path = 'pairs_scm128_before_nooverlap.json'
import mxnet as mx
import numpy as np
import cv2
from mxnet import nd
from functools import lru_cache
from sklearn import decomposition
from time import time

from retrieval.dataloader.transform import transform_test
from retrieval.mxsymbol import symbol_factory


class Retrieval_model():
    def __init__(self, ctx):
        self.DEBUG = False
        self.prefix = 'retrieval'
        #self.pca = decomposition.PCA(n_components=128)
        self.ctx = ctx
        self.model = self.get_mod(folder_name='retrieval/checkpoint', checkpoint_name='net_best.params', ctx=ctx)
        # self.anchors_data, self.cropus_data = self.load_search_database('database_sm128')
        self.database_names = ['type3_skirt','type2_skirt' ]
        self.database = self.load_search_database(self.database_names)

        #(self.cropus_index, self.cropus_index_inverse) = self.load_search_index(self.database_names)
        self.cropus_index_inverse = self.load_search_index(self.database_names)
        self.cropus_hist = self.load_hist_database(self.database_names)

    def load_hist_database(self,kinds):
        res = {}
        for kind in kinds:
            base = 'retrieval/cropus/hist'
            res[kind] = np.load(osp.join(base,kind,'cropus_hist.npy'))
        return res

    def get_hist(self, img):
        if not isinstance(img,nd.NDArray):
            img1 = nd.array(img)
        img1 = mx.img.resize_short(img1,224)
        img1 = mx.img.center_crop(img1,(112,112))[0].asnumpy().astype(np.uint8)
        #logger.info(img.dtype)
        #logger.info(img1.dtype)
        #logger.info(img1.shape)
        #logger.info(img.shape)
        #cv2.imwrite('hist_test_img.jpg',img)
        #cv2.imwrite('hist_test_img1.jpg',img1)
        img = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
        return np.mean(np.transpose(img, (2, 0, 1)), axis=(1, 2))

    def compare_histgram(self, h1, h2):
        #logger.info('h1')
        #logger.info(h1)
        #logger.info('h2')
        #logger.info(h2)
        return np.mean(np.abs(h1 - h2))

    def load_search_database(self, kinds):
        res = {}
        for kind in kinds:
            cropus_datapath = osp.join('retrieval/cropus', kind, 'cropus1920.npy')
            assert osp.exists(cropus_datapath), 'No cropus {}'.format(cropus_datapath)
            res[kind] = np.load(cropus_datapath)
        return res

    def load_search_index(self,kinds):
        res = {}
        def get_index(fn):
            with open(fn, 'r') as f:
                f_line = list(map(lambda x: x.strip(), f.readlines()))
            d = {}
            for idx, i in enumerate(f_line):
                d[i] = idx
            return d, f_line
        for kind in kinds:
            p = 'retrieval/cropus/lst'
            cropus_datapath = osp.join(p,kind,'cropus.lst')
            res[kind] = get_index(cropus_datapath)
        return res



    def get_feature(self, img):
        if not isinstance(img,nd.NDArray):
            img = nd.array(img)
        cv2.imwrite('test.jpg',img.asnumpy())
        imgs = transform_test(img)
        fea = nd.mean(self.model(imgs),axis=0)
        '''
        fea = \
        logger.info('h1')
        logger.info(h1)
            nd.mean(nd.stack(*[self.model(transform_val(img, None)[0].expand_dims(axis=0)) for i in range(tt)]),
                    axis=0)[0]
        '''
        #print(np.sum(np.abs(fea.asnumpy())))
        #fea = self.pca.transform(fea.asnumpy()[np.newaxis,])

        #fea = self.pca.transform(fea.asnumpy().reshape(1,-1))
        #print(np.sum(np.abs(fea)))
        return fea.asnumpy().reshape(1,-1)

    def search_database(self, img, color_level,style_level,database):
        threshold_styles = [0.24,0.38,0.28]
        threshold_style = threshold_styles[style_level]
        color_styles = [6,8]
        color_style = color_styles[color_level]
        anchor = self.get_feature(img)
        anchor_hist = self.get_hist(img)
        temp = np.sum((self.database[database] - anchor) ** 2, axis=1)
        res = list(map(lambda x: x, filter(lambda x: temp[x] < threshold_style, np.argsort(temp)[:2048])))
        logger.info(sorted(temp)[:64])
        cropus_hists = self.cropus_hist[database][res]
        distance = np.arange(cropus_hists.shape[0]).astype(np.float32)
        for idx, i in enumerate(cropus_hists):
            distance[idx] = self.compare_histgram(anchor_hist, i)
        temp = []
        temp1 = []
        for idx, i in enumerate(res):
            if distance[idx] < color_style:
                temp.append(i)
                '''
                elif distance[idx] < 15:
                    temp1.append(i)
                '''
        res = list(map(lambda x: self.cropus_index_inverse[database][x], temp[:]))[:64]
        if len(res) == 0:
            logger.info('no result')
        elif len(res) < 10:
            logger.info('result counts less than 10')
        return res

    def get_mod(self, folder_name, checkpoint_name, ctx=mx.cpu()):
        net = symbol_factory.get_test_symbol(ctx=ctx)
        net.load_params(osp.join(folder_name, checkpoint_name), ctx=ctx)
        net.hybridize()
        return net


if __name__ == '__main__':
    model = Retrieval_model(ctx=mx.cpu())
    pairs_json = {}
    tic = time()
    # print(model.search('820115'))
    # imglist = read_to_list(test_txt_path)

    tic = time()
    fea1 = model.get_feature('../demo/2.png', tt=1)
    print(time() - tic)
    import time

    time.sleep(1000)
    '''
    print(nd.sum(fea1 == 0))
    fea2 = nd.stack(*[model.get_feature(osp.join('../demo', i)) for i in
            ['177838.jpg', '202027.jpg','377045.jpg','526853.jpg','img.jpeg','img1.jpg','1.png','img2.jpg','3.png']])
    print(nd.argsort(nd.sum((fea1 - fea2) ** 2,axis=1)))
    '''
