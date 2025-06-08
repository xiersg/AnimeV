"""
图片预处理脚本 - 自动缩放角色部件图片
功能：
  1. 扫描原始图片目录
  2. 将所有图片缩放到统一的分辨率
  3. 保持透明通道
  4. 保存到新目录
"""

import os
import cv2
import numpy as np
from PIL import Image
from argparses import file_path_get

# 配置参数
INPUT_DIR = file_path_get()
OUTPUT_DIR = "processed_character_parts"  # 处理后图片目录
OUTPUT_DIR = os.path.join(OUTPUT_DIR,INPUT_DIR)
TARGET_SIZE = (200, 200)  # 目标分辨率 (宽, 高)
BACKGROUND_COLOR = (0, 0, 0, 0)  # 透明背景 (RGBA)

def process_images(INPUt_DIR):
    """处理所有图片"""
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    # 获取所有图片文件
    image_files = [f for f in os.listdir(INPUT_DIR)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"在 {INPUT_DIR} 目录中未找到图片文件")
        return

    print(f"找到 {len(image_files)} 个图片文件, 开始处理...")

    # 处理每张图片
    for i, filename in enumerate(image_files):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        try:
            # 使用PIL处理透明通道
            img = Image.open(input_path)

            # 转换为RGBA模式（确保有透明通道）
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # 创建新图像（透明背景）
            new_img = Image.new('RGBA', TARGET_SIZE, BACKGROUND_COLOR)

            # 计算缩放比例并居中放置
            width, height = img.size
            scale = min(TARGET_SIZE[0] / width, TARGET_SIZE[1] / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)

            # 计算位置（居中）
            position = (
                (TARGET_SIZE[0] - new_width) // 2,
                (TARGET_SIZE[1] - new_height) // 2
            )

            # 合并图像
            new_img.paste(img_resized, position, img_resized)

            # 保存处理后的图片
            new_img.save(output_path)
            print(f"处理完成 ({i+1}/{len(image_files)}): {filename} => {new_width}x{new_height}")

        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")

    print("所有图片处理完成!")

if __name__ == "__main__":
    # 检查输入目录
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录 {INPUT_DIR} 不存在")
        print("请创建目录并放入角色部件图片")
        os.makedirs(INPUT_DIR)
        print(f"已创建目录: {INPUT_DIR}")
    else:
        process_images(INPUT_DIR)
        print("退出")