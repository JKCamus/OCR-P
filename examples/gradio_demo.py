# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 改进的OCR演示界面
"""

import os
import gradio as gr
import numpy as np
from PIL import Image
from imgocr.ppocr_onnx import ImgOcr, draw_ocr_boxes
from imgocr.image_loader import load_image

# 初始化模型
efficiency_model = ImgOcr(use_gpu=False, is_efficiency_mode=True)
accuracy_model = None  # 延迟加载高精度模型，节省内存

def process_image(img_path, model_type="efficiency"):
    """处理图像并返回OCR结果和可视化图像"""
    global accuracy_model
    
    # 根据选择使用不同的模型
    if model_type == "accuracy" and accuracy_model is None:
        # 首次使用高精度模型时加载
        accuracy_model = ImgOcr(use_gpu=False, is_efficiency_mode=False)
    
    model = efficiency_model if model_type == "efficiency" else accuracy_model
    
    # 执行OCR识别
    ocr_result = model.ocr(img_path)
    
    # 提取文本结果
    ocr_text = [f"{i['text']} (置信度: {i['score']:.2f})" for i in ocr_result]
    text_result = '\n'.join(ocr_text)
    
    # 创建可视化图像
    img = load_image(img_path)
    boxes = [res['box'] for res in ocr_result]
    txts = [res['text'] for res in ocr_result]
    scores = [res['score'] for res in ocr_result]
    
    # 使用draw_ocr_boxes绘制结果
    vis_img = draw_ocr_boxes(img, ocr_result)
    
    # 统计信息
    stats = f"识别到 {len(ocr_result)} 个文本区域\n"
    stats += f"平均置信度: {np.mean(scores):.4f}\n"
    
    return text_result, vis_img, stats

def create_demo():
    with gr.Blocks(
        theme=gr.themes.Soft(), 
        title="imgocr - 中英文OCR识别工具",
        css="""
        footer {display: none !important;}
        .gradio-container {min-height: 0px !important;}
        """
    ) as demo:
        gr.Markdown(
            """
            # 📝 imgocr: 中英文OCR识别工具
            
            基于PaddleOCR-v4-onnx模型（~14MB）推理，性能更高，可实现 CPU 上毫秒级的 OCR 精准预测。
            

            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入部分
                input_image = gr.Image(type="filepath", label="上传图片")
                model_choice = gr.Radio(
                    ["efficiency", "accuracy"], 
                    label="模型选择", 
                    value="efficiency",
                    info="高效率模型(14MB)速度更快，高精度模型(207MB)准确率更高"
                )
                process_btn = gr.Button("开始识别", variant="primary")
                
                # 统计信息
                stats_output = gr.Textbox(label="统计信息", lines=3)
                
            with gr.Column(scale=2):
                # 输出部分
                with gr.Tab("文本结果"):
                    text_output = gr.Textbox(label="识别结果", lines=10)
                with gr.Tab("可视化结果"):
                    image_output = gr.Image(label="标注图像")
        
        # 示例图片
        example_dir = "examples/data"
        examples = [
            [os.path.join(example_dir, "11.jpg")],
            [os.path.join(example_dir, "00111002.jpg")],
            [os.path.join(example_dir, "00015504.jpg")],
            [os.path.join(example_dir, "eng_paper.png")],
        ]
        gr.Examples(examples=examples, inputs=input_image)
        
        # 处理逻辑
        process_btn.click(
            fn=process_image,
            inputs=[input_image, model_choice],
            outputs=[text_output, image_output, stats_output]
        )
        
        # 自动处理上传的图片
        input_image.change(
            fn=process_image,
            inputs=[input_image, model_choice],
            outputs=[text_output, image_output, stats_output]
        )
        
        # 更新说明
        gr.Markdown(
            """
            ### 使用说明
            
            1. 上传图片或选择示例图片
            2. 选择模型类型（高效率或高精度）
            3. 点击"开始识别"按钮（或等待自动处理）
            4. 查看"文本结果"和"可视化结果"标签页
            
            ### 注意事项
            
            - 首次使用高精度模型时需要下载模型文件（约207MB），可能需要等待一段时间
            - 支持中文、英文、数字等多语种识别
            """
        )
    
    return demo

if __name__ == '__main__':
    import signal
    import os
    import atexit
    import threading
    
    # 记录当前进程ID，用于清理
    current_pid = os.getpid()
    
    # 创建output目录（如果不存在）用于保存输出文件
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义清理函数
    def cleanup_handler():
        print("\n清理资源并退出程序...")
        # 可以在这里添加额外的清理代码
    
    # 定义信号处理函数
    def signal_handler(sig, frame):
        print(f"\n接收到信号 {sig}，正在退出...")
        cleanup_handler()
        # 强制终止当前进程
        os._exit(0)
    
    # 注册清理函数，在正常退出时调用
    atexit.register(cleanup_handler)
    
    # 注册信号处理函数，捕获常见的终止信号
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # kill命令
    
    # 创建并启动Gradio界面
    try:
        demo = create_demo()
        # 使用兼容的参数启动Gradio
        demo.launch(share=False, debug=True, prevent_thread_lock=True)
        
        # 主线程等待，保持对信号的响应
        while threading.active_count() > 0:
            try:
                # 使主线程保持活跃但允许接收信号
                threading.Event().wait(1)
            except KeyboardInterrupt:
                # 捕获Ctrl+C
                signal_handler(signal.SIGINT, None)
                break
            
    except Exception as e:
        print(f"发生错误: {e}")
        cleanup_handler()
        os._exit(1)
