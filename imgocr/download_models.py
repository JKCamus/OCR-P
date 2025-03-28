#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@description: 自动下载OCR模型文件
"""

import os
import requests
from tqdm import tqdm
import hashlib
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型文件信息
MODEL_INFO = {
    # 移动端模型（高效率）
    "ch_PP-OCRv4_det_mobile_infer.onnx": {
        "url": "https://huggingface.co/lili666/imgocr/resolve/main/ch_PP-OCRv4_det_mobile_infer.onnx",
        "md5": "d4e7b8667e2e1b1b5f05c21e3b4b4a4e",  # 示例MD5，需要替换为实际值
        "size": 2.5,  # 大小（MB）
    },
    "ch_PP-OCRv4_rec_mobile_infer.onnx": {
        "url": "https://huggingface.co/lili666/imgocr/resolve/main/ch_PP-OCRv4_rec_mobile_infer.onnx",
        "md5": "a7b0e46e1e4c1a6f2f0a9a9e5b8b8e8d",  # 示例MD5，需要替换为实际值
        "size": 11.5,  # 大小（MB）
    },
    "ch_PP-OCRv4_cls_infer.onnx": {
        "url": "https://huggingface.co/lili666/imgocr/resolve/main/ch_PP-OCRv4_cls_infer.onnx",
        "md5": "c1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6",  # 示例MD5，需要替换为实际值
        "size": 1.5,  # 大小（MB）
    },
    
    # 服务器端模型（高精度）
    "ch_PP-OCRv4_det_server_infer.onnx": {
        "url": "https://huggingface.co/lili666/imgocr/resolve/main/ch_PP-OCRv4_det_server_infer.onnx",
        "md5": "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d",  # 示例MD5，需要替换为实际值
        "size": 108.1,  # 大小（MB）
    },
    "ch_PP-OCRv4_rec_server_infer.onnx": {
        "url": "https://huggingface.co/lili666/imgocr/resolve/main/ch_PP-OCRv4_rec_server_infer.onnx",
        "md5": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",  # 示例MD5，需要替换为实际值
        "size": 86.3,  # 大小（MB）
    },
}

def get_md5(file_path):
    """计算文件的MD5值"""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()

def download_file(url, save_path, expected_md5=None):
    """下载文件并显示进度条"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 创建进度条
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(save_path))
        
        # 下载文件
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        
        # 验证MD5
        if expected_md5:
            actual_md5 = get_md5(save_path)
            if actual_md5 != expected_md5:
                logger.warning(f"MD5校验失败: {save_path}")
                logger.warning(f"预期: {expected_md5}, 实际: {actual_md5}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"下载失败: {url}")
        logger.error(f"错误信息: {e}")
        return False

def download_models(model_dir, mode='efficiency'):
    """下载模型文件
    
    Args:
        model_dir: 模型保存目录
        mode: 'efficiency'(高效率模型) 或 'all'(全部模型)
    """
    # 创建模型目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 确定要下载的模型
    models_to_download = []
    
    if mode == 'efficiency':
        # 只下载移动端模型
        models_to_download = [
            "ch_PP-OCRv4_det_mobile_infer.onnx",
            "ch_PP-OCRv4_rec_mobile_infer.onnx",
            "ch_PP-OCRv4_cls_infer.onnx"
        ]
    else:  # 'all'
        # 下载所有模型
        models_to_download = list(MODEL_INFO.keys())
    
    # 下载模型
    success_count = 0
    for model_name in models_to_download:
        model_path = os.path.join(model_dir, model_name)
        
        # 检查文件是否已存在且MD5正确
        if os.path.exists(model_path):
            if MODEL_INFO[model_name].get("md5"):
                actual_md5 = get_md5(model_path)
                if actual_md5 == MODEL_INFO[model_name]["md5"]:
                    logger.info(f"模型已存在且MD5正确: {model_name}")
                    success_count += 1
                    continue
            else:
                logger.info(f"模型已存在: {model_name}")
                success_count += 1
                continue
        
        # 下载模型
        logger.info(f"开始下载模型: {model_name} (约 {MODEL_INFO[model_name]['size']} MB)")
        success = download_file(
            MODEL_INFO[model_name]["url"],
            model_path,
            MODEL_INFO[model_name].get("md5")
        )
        
        if success:
            logger.info(f"模型下载成功: {model_name}")
            success_count += 1
        else:
            logger.error(f"模型下载失败: {model_name}")
    
    # 返回下载结果
    return success_count == len(models_to_download)

def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")
    
    # 解析命令行参数
    mode = 'efficiency'  # 默认只下载高效率模型
    if len(sys.argv) > 1 and sys.argv[1] == 'all':
        mode = 'all'
    
    # 下载模型
    logger.info(f"开始下载模型 (模式: {mode})...")
    success = download_models(model_dir, mode)
    
    if success:
        logger.info("所有模型下载成功!")
    else:
        logger.error("部分模型下载失败，请检查网络连接后重试。")
        sys.exit(1)

if __name__ == "__main__":
    main()
