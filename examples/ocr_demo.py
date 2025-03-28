# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Image ocr demo.
"""
import time
import sys
import os

sys.path.append('..')
from imgocr.ppocr_onnx import ImgOcr
from imgocr.ppocr_onnx import draw_ocr_boxes

if __name__ == "__main__":
    m = ImgOcr(use_gpu=False, is_efficiency_mode=True)
    img_path = "examples/data/11.jpg"
    s = time.time()
    result = m.ocr(img_path)
    e = time.time()
    print("total time: {:.4f} s".format(e - s))
    print("result:", result)
    for i in result:
        print(i['text'])

    # 创建output目录（如果不存在）
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    output_file = os.path.join(output_dir, '11_box.jpg')
    
    # draw boxes
    draw_ocr_boxes(img_path, result, output_file)
    print(f'Save result to {output_file}')
