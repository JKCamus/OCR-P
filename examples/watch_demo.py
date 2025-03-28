#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动监视文件变化并重启Gradio应用程序的脚本
"""

import os
import sys
import time
import subprocess
import signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 要监视的目录
WATCH_DIRS = ['examples', 'imgocr']
# 要监视的文件扩展名
WATCH_EXTENSIONS = ['.py']
# 要运行的脚本
TARGET_SCRIPT = 'examples/gradio_demo.py'

class ChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.restart_process()
        
    def restart_process(self):
        """终止当前进程并启动新进程"""
        if self.process:
            print("\n检测到文件变化，正在重启应用...")
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process = None
        
        # 启动新进程
        print(f"\n启动 {TARGET_SCRIPT}...")
        self.process = subprocess.Popen(
            f"python {TARGET_SCRIPT}", 
            shell=True, 
            preexec_fn=os.setsid
        )
    
    def on_modified(self, event):
        """文件被修改时触发"""
        if not event.is_directory and any(event.src_path.endswith(ext) for ext in WATCH_EXTENSIONS):
            print(f"文件已修改: {event.src_path}")
            self.restart_process()
    
    def on_created(self, event):
        """文件被创建时触发"""
        if not event.is_directory and any(event.src_path.endswith(ext) for ext in WATCH_EXTENSIONS):
            print(f"文件已创建: {event.src_path}")
            self.restart_process()

def main():
    # 创建事件处理器和观察者
    event_handler = ChangeHandler()
    observer = Observer()
    
    # 为每个目录添加监视
    for directory in WATCH_DIRS:
        path = os.path.join(os.getcwd(), directory)
        if os.path.exists(path):
            print(f"监视目录: {path}")
            observer.schedule(event_handler, path, recursive=True)
    
    # 启动观察者
    observer.start()
    print(f"热更新已启动，监视 {', '.join(WATCH_DIRS)} 目录中的 {', '.join(WATCH_EXTENSIONS)} 文件...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # 终止应用程序进程
        if event_handler.process:
            os.killpg(os.getpgid(event_handler.process.pid), signal.SIGTERM)
        # 停止观察者
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    main()
