#
# CRNN文字识别：速度不错
#
# Q:
# 1. 利用ctpn识别文字区域后，再逐一用crnn识别文字
# 2. 对较瘦高的文字识别率低：
# 3. 如何用自己的数据训练？
#

# coding:utf-8

import dataset
import keys_crnn
import models.crnn as crnn
import torch.utils.data
import util
import os
from PIL import Image
from torch.autograd import Variable

alphabet = keys_crnn.alphabet
print("可识别字符数：", len(alphabet))
# raw_input('\ninput:')
converter = util.strLabelConverter(alphabet)
# model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1)
# path = './samples/netCRNN63.pth'
path = './samples/model_acc97.pth'
model.load_state_dict(torch.load(path))
# print(model)

fs = os.listdir("../img/")
for f in fs:
    img_file = os.path.join("../img/", f)
    image = Image.open(img_file).convert('L')
    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    print(img_file)

    transformer = dataset.resizeNormalize((w, 32))
    # image = transformer(image).cuda()
    image = transformer(image)
    image = image.view(1, *image.size())
    image = Variable(image)
    model.eval()
    preds = model(image)
    _, preds = preds.max(2)
    # preds = preds.squeeze(2)          # 作用？？
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    # raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print('%-20s =>' % (raw_pred))
    print('%-20s' % (sim_pred))
