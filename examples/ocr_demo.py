# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Image ocr demo.
"""
import time
import sys

sys.path.append('..')
from imgocr import ImgOcr
from imgocr import drwa_ocr_boxes

if __name__ == "__main__":
    m = ImgOcr(use_gpu=False, is_efficiency_mode=True)
    img_path = "data/11.jpg"
    s = time.time()
    result = m.ocr(img_path)
    e = time.time()
    print("total time: {:.4f} s".format(e - s))
    print("result:", result)
    for line in result[0]:
        print(line)

    # draw boxes
    drwa_ocr_boxes(img_path, result, '11_box.jpg')
    print('Save result to 11_box.jpg')
