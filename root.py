from flask import request
from flask import Flask
from model import model
from logger import logger

app = Flask(__name__)


@app.route('/10001', methods=['GET', 'POST'])
def zero():
    js = ((request.get_json()))
    js['imgId'] = '0'
    logger.info(js)
    res = str(model.do_multi_predict(image_url=js['url'], model_level=int(10001), img_id=str(js['imgId']), tt=2))
    logger.info(res)
    return res


@app.route('/1', methods=['GET', 'POST'])
def first():
    js = ((request.get_json()))
    js['imgId'] = '0'
    logger.info(js)
    res = str(model.do_multi_predict(image_url=js['url'], model_level=int(1), img_id=str(js['imgId']), tt=2))
    logger.info(res)
    return res


@app.route('/2', methods=['GET', 'POST'])
def second():
    js = ((request.get_json()))
    js['imgId'] = '0'
    logger.info(js)
    if str(js['class1']) not in ['1','4','5','7','10']:
        return 'model not trained yet'
    res = str(model.do_multi_predict(image_url=js['url'], model_level=int(2), first_class_id=int(js['class1']),
                                     img_id=str(js['imgId']), tt=3))
    logger.info(res)
    return res


@app.route('/10003', methods=['GET', 'POST'])
def retrieval():
    js = ((request.get_json()))
    js['imgId'] = '0'
    logger.info(js)
    if js['color_level'] not in [0,1,2] and js['style_level'] not in [0,1,2]:
        return 'selected level wrong'
    res = str(model.retrieval(image_url=js['url'],color_level=int(js['color_level']),style_level=int(js['style_level']),database='type3_skirt'))
    logger.info(res)
    return res

@app.route('/10004', methods=['GET', 'POST'])
def retrieval_temp():
    js = ((request.get_json()))
    js['imgId'] = '0'
    logger.info(js)
    if js['color_level'] not in [0,1,2] and js['style_level'] not in [0,1,2]:
        return 'selected level wrong'
    res = str(model.retrieval(image_url=js['url'],color_level=int(js['color_level']),style_level=int(js['style_level'])))
    logger.info(res)
    return res

