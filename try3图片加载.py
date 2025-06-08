"""
修复版骨骼绑定二次元人物驱动系统
修复了 PoseLandmark 实例化错误
移除了颈部连接，保持简单稳定
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join("units")
sys.path.append(scripts_dir)
import argparse
from argparses import file_path_get

#获取地址
path = file_path_get()


# 初始化MediaPipe姿态检测
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=2
)

# 初始化Pygame
pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("骨骼绑定二次元人物驱动 - 按ESC退出")
clock = pygame.time.Clock()

# ======================== 人物资源加载 ========================
# 创建资源目录
resource_dir = path
if not os.path.exists(resource_dir):
    os.makedirs(resource_dir)
    print(f"请将人物部件图片放入 {resource_dir} 目录")

# 身体部件定义（根据实际图片调整锚点位置）
BODY_PARTS = {
    "head": {"file": "head.png", "anchor": (0.5, 0.8)},  # 锚点位置 (比例x, 比例y)
    "body": {"file": "body.png", "anchor": (0.5, 0.2)},
    "left_upper_arm": {"file": "left_upper_arm.png", "anchor": (0.2, 0.5)},
    "left_lower_arm": {"file": "left_lower_arm.png", "anchor": (0.2, 0.8)},
    "left_hand": {"file": "left_hand.png", "anchor": (0.5, 0.2)},
    "right_upper_arm": {"file": "right_upper_arm.png", "anchor": (0.8, 0.5)},
    "right_lower_arm": {"file": "right_lower_arm.png", "anchor": (0.8, 0.8)},
    "right_hand": {"file": "right_hand.png", "anchor": (0.5, 0.2)},
    "left_upper_leg": {"file": "left_upper_leg.png", "anchor": (0.3, 0.2)},
    "left_lower_leg": {"file": "left_lower_leg.png", "anchor": (0.5, 0.2)},
    "left_foot": {"file": "left_foot.png", "anchor": (0.5, 0.2)},
    "right_upper_leg": {"file": "right_upper_leg.png", "anchor": (0.7, 0.2)},
    "right_lower_leg": {"file": "right_lower_leg.png", "anchor": (0.5, 0.2)},
    "right_foot": {"file": "right_foot.png", "anchor": (0.5, 0.2)}
}

# 加载身体部件图片
character_parts = {}
for part, info in BODY_PARTS.items():
    try:
        img = pygame.image.load(os.path.join(resource_dir, info["file"]))
        img = img.convert_alpha()  # 保留透明通道
        character_parts[part] = {
            "image": img,
            "anchor": info["anchor"],
            "rect": img.get_rect()
        }
        print(f"加载部件: {part} 尺寸: {img.get_size()}")
    except:
        print(f"警告: 无法加载部件 {info['file']}, 将使用占位图形")
        # 创建占位图形
        surf = pygame.Surface((50, 50), pygame.SRCALPHA)
        pygame.draw.rect(surf, (255, 0, 0, 128), (0, 0, 50, 50))
        character_parts[part] = {
            "image": surf,
            "anchor": (0.5, 0.5),
            "rect": surf.get_rect()
        }

# 部件绑定关系（MediaPipe关键点索引）
PART_BINDINGS = {
    "head": (mp_pose.PoseLandmark.NOSE, None),  # 头部使用鼻子位置
    "body": (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    "left_upper_arm": (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    "left_lower_arm": (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    "left_hand": (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_PINKY),
    "right_upper_arm": (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    "right_lower_arm": (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    "right_hand": (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_PINKY),
    "left_upper_leg": (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    "left_lower_leg": (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    "left_foot": (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL),
    "right_upper_leg": (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    "right_lower_leg": (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    "right_foot": (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL)
}

# 渲染顺序（从后到前）
RENDER_ORDER = [
    "left_upper_leg", "right_upper_leg",
    "left_lower_leg", "right_lower_leg",
    "left_foot", "right_foot",
    "body",
    "left_upper_arm", "right_upper_arm",
    "left_lower_arm", "right_lower_arm",
    "left_hand", "right_hand",
    "head"
]

# 背景颜色
BACKGROUND_COLOR = (240, 248, 255)

def calculate_rotation(start_point, end_point):
    """
    计算两点之间的旋转角度
    参数:
        start_point: 起始点 (x, y)
        end_point: 结束点 (x, y)
    返回:
        angle: 旋转角度（度）
    """
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    # 注意：Pygame的y轴向下为正，所以角度计算需要调整
    angle = math.degrees(math.atan2(-dy, dx))
    return angle

def transform_part(part, landmarks, binding, scale=1.0):
    """
    变换身体部件
    参数:
        part: 部件信息
        landmarks: 所有关键点
        binding: 绑定关系 (start_index, end_index)
        scale: 缩放比例
    返回:
        transformed_image: 变换后的图像
        position: 位置 (x, y)
    """
    # 获取绑定的关键点
    start_idx = binding[0].value
    end_idx = binding[1].value if binding[1] else None

    # 获取起始点位置
    start_point = landmarks[start_idx]
    start_x = int(start_point["x"] * WIDTH)
    start_y = int(start_point["y"] * HEIGHT)

    # 计算旋转角度
    angle = 0
    if end_idx is not None and end_idx < len(landmarks):
        end_point = landmarks[end_idx]
        end_x = int(end_point["x"] * WIDTH)
        end_y = int(end_point["y"] * HEIGHT)
        angle = calculate_rotation((start_x, start_y), (end_x, end_y))

    # 获取原始图像
    original_image = part["image"]

    # 缩放图像
    if scale != 1.0:
        new_width = int(original_image.get_width() * scale)
        new_height = int(original_image.get_height() * scale)
        scaled_image = pygame.transform.scale(original_image, (new_width, new_height))
    else:
        scaled_image = original_image

    # 旋转图像
    rotated_image = pygame.transform.rotate(scaled_image, -angle)  # 负号因为旋转方向

    # 计算锚点偏移
    anchor_x = int(part["anchor"][0] * scaled_image.get_width())
    anchor_y = int(part["anchor"][1] * scaled_image.get_height())

    # 计算旋转后的位置
    rotated_rect = rotated_image.get_rect()
    pos_x = start_x - rotated_rect.width // 2
    pos_y = start_y - rotated_rect.height // 2

    return rotated_image, (pos_x, pos_y)

def draw_character(screen, landmarks):
    """
    绘制骨骼绑定的人物
    参数:
        screen: Pygame屏幕对象
        landmarks: 关键点列表
    """
    if not landmarks:
        return

    # 绘制所有部件（按顺序）
    for part_name in RENDER_ORDER:
        if part_name in character_parts and part_name in PART_BINDINGS:
            part = character_parts[part_name]
            binding = PART_BINDINGS[part_name]

            # 获取变换后的部件
            img, pos = transform_part(part, landmarks, binding)

            # 绘制部件
            screen.blit(img, pos)

def process_frame(frame):
    """处理摄像头帧并检测姿态关键点"""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    return results.pose_landmarks.landmark if results.pose_landmarks else None

# ======================== 主程序 ========================
# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 加载字体
try:
    font = pygame.font.SysFont("microsoftyahei", 24)
except:
    font = pygame.font.SysFont(None, 24)

# 人物位置偏移
character_offset_x = WIDTH // 2
character_offset_y = HEIGHT // 2

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_UP:
                character_offset_y -= 10
            elif event.key == pygame.K_DOWN:
                character_offset_y += 10
            elif event.key == pygame.K_LEFT:
                character_offset_x -= 10
            elif event.key == pygame.K_RIGHT:
                character_offset_x += 10

    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("无法从摄像头获取帧")
        continue

    # 水平镜像翻转
    frame = cv2.flip(frame, 1)

    # 处理帧并获取关键点
    landmarks = process_frame(frame)

    # 清空屏幕
    screen.fill(BACKGROUND_COLOR)

    # 绘制人物
    if landmarks:
        # 创建临时的偏移关键点 - 修复版本
        # 使用简单的字典来存储关键点信息
        offset_landmarks = []
        for lm in landmarks:
            # 复制关键点属性
            offset_landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "visibility": lm.visibility
            })

        # 调整位置（缩放并居中）
        for lm in offset_landmarks:
            lm["x"] = lm["x"] * 0.7 + 0.15
            lm["y"] = lm["y"] * 0.7 + 0.15

        # 绘制人物
        draw_character(screen, offset_landmarks)

    # 绘制说明文本
    title = font.render("骨骼绑定二次元人物驱动系统", True, (0, 0, 0))
    help_text = font.render("方向键移动人物位置 | ESC退出", True, (100, 100, 100))
    screen.blit(title, (20, 20))
    screen.blit(help_text, (20, HEIGHT - 40))

    # 显示帧率
    fps_text = font.render(f"帧率: {int(clock.get_fps())} FPS", True, (0, 0, 0))
    screen.blit(fps_text, (WIDTH - 150, HEIGHT - 40))

    # 更新显示
    pygame.display.flip()
    clock.tick(30)

# 清理资源
cap.release()
pygame.quit()