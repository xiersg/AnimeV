"""
双视图动作对比系统
功能：
  左侧：显示摄像头原始画面，并绘制MediaPipe检测到的节点和骨骼
  右侧：显示同步动作的二次元人物
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import sys

# 初始化MediaPipe姿态检测
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # 用于绘制原始节点
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=2
)

# 初始化Pygame
pygame.init()
TOTAL_WIDTH, HEIGHT = 1600, 600  # 总宽度=原始画面宽度+二次元画面宽度
screen = pygame.display.set_mode((TOTAL_WIDTH, HEIGHT))
pygame.display.set_caption("双视图动作对比系统 - 按ESC退出")
clock = pygame.time.Clock()

# ======================== 二次元人物参数配置 ========================
SKIN_COLOR = (255, 224, 189)
HAIR_COLOR = (60, 40, 20)
CLOTHES_COLOR = (100, 150, 255)
LINE_COLOR = (50, 50, 50)
BACKGROUND_COLOR = (240, 248, 255)
SECONDARY_BG_COLOR = (220, 230, 245)  # 二次元区域的背景色

# 人物大小参数
HEAD_RADIUS = 40
JOINT_RADIUS = 8
BONE_THICKNESS = 3
EYE_RADIUS = 8
PUPIL_RADIUS = 4

# 视图区域定义
CAMERA_WIDTH = 640  # 摄像头视图宽度
CAMERA_HEIGHT = 480  # 摄像头视图高度
ANIME_WIDTH = TOTAL_WIDTH - CAMERA_WIDTH  # 二次元视图宽度

# ======================== 骨骼连接关系定义 ========================
CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    (mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.LEFT_EYE_OUTER),
    (mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.LEFT_EYE),
    (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.NOSE),
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EYE),
    (mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER),
    (mp_pose.PoseLandmark.RIGHT_EYE_OUTER, mp_pose.PoseLandmark.RIGHT_EAR),
]


def draw_anime_character(screen, landmarks, connections, offset_x=0):
    """
    绘制二次元风格人物
    参数:
        screen: Pygame屏幕对象
        landmarks: 关键点列表
        connections: 骨骼连接关系
        offset_x: 绘制位置的X偏移量
    """
    if landmarks is None:
        return

    # 绘制身体连接线
    for connection in connections:
        start_idx = connection[0].value
        end_idx = connection[1].value

        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]

            if start_point.visibility > 0.3 and end_point.visibility > 0.3:
                # 添加偏移量并调整到二次元视图中心
                start_x = offset_x + ANIME_WIDTH // 2 + int((start_point.x - 0.5) * ANIME_WIDTH * 0.8)
                start_y = HEIGHT // 2 + int((start_point.y - 0.5) * HEIGHT * 0.8)
                end_x = offset_x + ANIME_WIDTH // 2 + int((end_point.x - 0.5) * ANIME_WIDTH * 0.8)
                end_y = HEIGHT // 2 + int((end_point.y - 0.5) * HEIGHT * 0.8)

                pygame.draw.line(screen, LINE_COLOR, (start_x, start_y), (end_x, end_y), BONE_THICKNESS)
                pygame.draw.circle(screen, SKIN_COLOR, (start_x, start_y), JOINT_RADIUS)
                pygame.draw.circle(screen, SKIN_COLOR, (end_x, end_y), JOINT_RADIUS)

    # 绘制头部
    if len(landmarks) > mp_pose.PoseLandmark.NOSE.value:
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        if nose.visibility > 0.3:
            head_x = offset_x + ANIME_WIDTH // 2 + int((nose.x - 0.5) * ANIME_WIDTH * 0.8)
            head_y = HEIGHT // 2 + int((nose.y - 0.5) * HEIGHT * 0.8) - 40

            pygame.draw.circle(screen, SKIN_COLOR, (head_x, head_y), HEAD_RADIUS)
            pygame.draw.ellipse(screen, HAIR_COLOR,
                                (head_x - HEAD_RADIUS * 1.25, head_y - HEAD_RADIUS * 1.25,
                                 HEAD_RADIUS * 2.5, HEAD_RADIUS * 1.5))

            # 眼睛
            eye_offset = 15
            pygame.draw.circle(screen, (255, 255, 255), (head_x - eye_offset, head_y - 5), EYE_RADIUS)
            pygame.draw.circle(screen, (0, 0, 0), (head_x - eye_offset, head_y - 5), PUPIL_RADIUS)
            pygame.draw.circle(screen, (255, 255, 255), (head_x + eye_offset, head_y - 5), EYE_RADIUS)
            pygame.draw.circle(screen, (0, 0, 0), (head_x + eye_offset, head_y - 5), PUPIL_RADIUS)

            # 嘴巴
            pygame.draw.arc(screen, (200, 100, 100),
                            (head_x - 20, head_y + 10, 40, 20),
                            0, math.pi, 2)


def draw_camera_frame(screen, frame, landmarks):
    """
    在Pygame屏幕上绘制摄像头帧和MediaPipe节点
    参数:
        screen: Pygame屏幕对象
        frame: OpenCV图像帧
        landmarks: MediaPipe检测到的关键点
    """
    # 将OpenCV图像转换为Pygame格式
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)  # 旋转90度
    frame = pygame.surfarray.make_surface(frame)

    # 绘制摄像头帧
    screen.blit(frame, (0, (HEIGHT - CAMERA_HEIGHT) // 2))

    # 如果检测到关键点，绘制节点和骨骼
    if landmarks:
        # 使用MediaPipe自带的绘图工具绘制骨骼
        for connection in CONNECTIONS:
            start_idx = connection[0].value
            end_idx = connection[1].value

            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]

                if start_point.visibility > 0.3 and end_point.visibility > 0.3:
                    # 转换坐标到屏幕位置
                    start_x = int(start_point.x * CAMERA_WIDTH)
                    start_y = int(start_point.y * CAMERA_HEIGHT) + (HEIGHT - CAMERA_HEIGHT) // 2
                    end_x = int(end_point.x * CAMERA_WIDTH)
                    end_y = int(end_point.y * CAMERA_HEIGHT) + (HEIGHT - CAMERA_HEIGHT) // 2

                    # 绘制骨骼
                    pygame.draw.line(screen, (0, 255, 0), (start_x, start_y), (end_x, end_y), 2)

                    # 绘制关节
                    pygame.draw.circle(screen, (255, 0, 0), (start_x, start_y), 5)
                    pygame.draw.circle(screen, (255, 0, 0), (end_x, end_y), 5)


def process_frame(frame):
    """处理摄像头帧并检测姿态关键点"""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    return results.pose_landmarks.landmark if results.pose_landmarks else None


# ======================== 主程序 ========================
# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# 加载字体
try:
    font = pygame.font.SysFont("microsoftyahei", 24)  # 使用微软雅黑
except:
    font = pygame.font.SysFont(None, 24)  # 回退到默认字体

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("无法从摄像头获取帧")
        continue

    # 处理帧并获取关键点
    landmarks = process_frame(frame)

    # 清空屏幕
    screen.fill(BACKGROUND_COLOR)

    # 绘制分隔线
    pygame.draw.line(screen, (100, 100, 100), (CAMERA_WIDTH, 0), (CAMERA_WIDTH, HEIGHT), 2)

    # 绘制摄像头视图
    frame_camera = cv2.flip(frame, 1)
    draw_camera_frame(screen, frame_camera, landmarks)

    # 绘制二次元人物视图
    # 填充二次元区域的背景色
    pygame.draw.rect(screen, SECONDARY_BG_COLOR, (CAMERA_WIDTH, 0, ANIME_WIDTH, HEIGHT))

    if landmarks:
        draw_anime_character(screen, landmarks, CONNECTIONS, offset_x=CAMERA_WIDTH)

    # 绘制标题
    camera_title = font.render("摄像头原始视图 (带MediaPipe节点)", True, (0, 0, 0))
    anime_title = font.render("二次元人物同步视图", True, (0, 0, 0))
    screen.blit(camera_title, (20, 20))
    screen.blit(anime_title, (CAMERA_WIDTH + 20, 20))

    # 绘制说明文本
    help_text = font.render("按ESC键退出程序", True, (100, 100, 100))
    screen.blit(help_text, (TOTAL_WIDTH - 200, HEIGHT - 40))

    # 显示帧率
    fps_text = font.render(f"帧率: {int(clock.get_fps())} FPS", True, (0, 0, 0))
    screen.blit(fps_text, (20, HEIGHT - 40))

    # 更新显示
    pygame.display.flip()
    clock.tick(30)

# 清理资源
cap.release()
pygame.quit()
sys.exit()