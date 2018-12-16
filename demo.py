#
# 能正常运行：可判断文字方向，多行识别，但速度太慢，识别准确率待提升（PyTorch的效果好些）！
#
# 1.将模型数据统一调整到ModelSet目录下；
# 2.训练自己的数据？

# coding:utf-8
import time
from glob import glob
import numpy as np
from PIL import Image

import model

paths = glob('./test/*.*')

if __name__ == '__main__':
    print("Demo ...")
    im = Image.open("./test/test.png")
    img = np.array(im.convert('RGB'))
    t = time.time()

    # result,img,angel分别对应-识别结果，图像的数组，文字旋转角度
    # result, img, angle = model.model(img, model='keras', adjust=True, detectAngle=True)
    result, img, angle = model.model(img, model='pytorch', adjust=True, detectAngle=True)

    print("It takes time:{}s".format(time.time() - t))
    print("---------------------------------------")
    for key in result:
        print(result[key][1])
