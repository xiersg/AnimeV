"""
摄像头驱动二次元人物动作同步实验
功能：通过摄像头捕捉真人动作，实时驱动二次元人物进行同步动作
使用库：
  - OpenCV: 用于摄像头捕获和图像处理
  - MediaPipe: 用于人体姿态关键点检测
  - Pygame: 用于渲染二次元风格人物
  - NumPy: 用于数学计算（本代码中未显式使用，但MediaPipe依赖它）
"""

import cv2  # 导入OpenCV库，用于摄像头捕获和图像处理
import mediapipe as mp  # 导入MediaPipe库，用于人体姿态检测
import pygame  # 导入Pygame库，用于创建图形界面和渲染
import numpy as np  # 导入NumPy库，用于数学计算（虽然本代码未直接使用，但MediaPipe依赖它）
import math  # 导入数学库，用于三角函数等计算

# 初始化MediaPipe的姿态检测模块
mp_pose = mp.solutions.pose  # 获取姿态检测模块
pose = mp_pose.Pose(  # 创建姿态检测对象a
    min_detection_confidence=0.5,  # 检测置信度阈值（0-1），值越高检测越严格
    min_tracking_confidence=0.5,  # 跟踪置信度阈值（0-1），值越高跟踪越稳定
    model_complexity=2  # 模型复杂度（0-2）：2为最高精度，使用33个关键点
)

# 初始化Pygame图形界面
pygame.init()  # 初始化Pygame
WIDTH, HEIGHT = 800, 600  # 设置窗口尺寸（宽度, 高度）
screen = pygame.display.set_mode((WIDTH, HEIGHT))  # 创建窗口
pygame.display.set_caption("二次元人物动作同步 - 按ESC退出")  # 设置窗口标题
clock = pygame.time.Clock()  # 创建时钟对象，用于控制帧率

# ======================== 二次元人物参数配置 ========================
# 你可以修改这些参数来自定义二次元人物的外观n
SKIN_COLOR = (255, 224, 189)  # 皮肤颜色 (R, G, B)
HAIR_COLOR = (60, 40, 20)  # 头发颜色
CLOTHES_COLOR = (100, 150, 255)  # 衣服颜色
LINE_COLOR = (50, 50, 50)  # 骨骼线条颜色
BACKGROUND_COLOR = (240, 248, 255)  # 背景颜色（浅蓝色）

# 人物大小参数（可调整）
HEAD_RADIUS = 40  # 头部半径
JOINT_RADIUS = 8  # 关节点的半径
BONE_THICKNESS = 3  # 骨骼线条粗细
EYE_RADIUS = 8  # 眼睛半径
PUPIL_RADIUS = 4  # 瞳孔半径

# ======================== 骨骼连接关系定义 ========================
# MediaPipe使用33个关键点，这里定义了哪些点应该连接形成骨骼
# 你可以修改这些连接关系来改变人物的骨骼结构
CONNECTIONS = [
    # 身体主干
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),  # 左右髋部连接
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER),  # 左髋到左肩
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER),  # 右髋到右肩
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),  # 左右肩连接

    # 左臂
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),  # 左肩到左肘
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),  # 左肘到左手腕

    # 右臂
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),  # 右肩到右肘
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),  # 右肘到右手腕

    # 左腿
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),  # 左髋到左膝
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),  # 左膝到左脚踝

    # 右腿
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),  # 右髋到右膝
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),  # 右膝到右脚踝

    # 头部（简化版）
    (mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.LEFT_EYE_OUTER),  # 左耳到左外眼角
    (mp_pose.PoseLandmark.LEFT_EYE_OUTER, mp_pose.PoseLandmark.LEFT_EYE),  # 左外眼角到左眼
    (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.NOSE),  # 左眼到鼻子
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EYE),  # 鼻子到右眼
    (mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EYE_OUTER),  # 右眼到右外眼角
    (mp_pose.PoseLandmark.RIGHT_EYE_OUTER, mp_pose.PoseLandmark.RIGHT_EAR),  # 右外眼角到右耳
]


def draw_anime_character(screen, landmarks, connections):
    """
    绘制二次元风格人物
    参数:
        screen: Pygame的屏幕对象
        landmarks: MediaPipe检测到的关键点列表
        connections: 骨骼连接关系定义
    """
    if landmarks is None:
        return  # 如果没有检测到关键点，直接返回

    # 1. 绘制骨骼连接线
    for connection in connections:
        start_idx = connection[0].value  # 获取起始点索引
        end_idx = connection[1].value  # 获取结束点索引

        # 确保索引在有效范围内
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = landmarks[start_idx]  # 起始点对象
            end_point = landmarks[end_idx]  # 结束点对象

            # 检查关键点可见性（值0-1，越高表示越可靠）
            if start_point.visibility > 0.3 and end_point.visibility > 0.3:
                # 将归一化坐标转换为屏幕坐标
                start_x = int(start_point.x * WIDTH)
                start_y = int(start_point.y * HEIGHT)
                end_x = int(end_point.x * WIDTH)
                end_y = int(end_point.y * HEIGHT)

                # 绘制骨骼线条
                pygame.draw.line(screen, LINE_COLOR, (start_x, start_y), (end_x, end_y), BONE_THICKNESS)

                # 在关节处绘制圆形
                pygame.draw.circle(screen, SKIN_COLOR, (start_x, start_y), JOINT_RADIUS)
                pygame.draw.circle(screen, SKIN_COLOR, (end_x, end_y), JOINT_RADIUS)

    # 2. 绘制头部 (二次元风格)
    if len(landmarks) > mp_pose.PoseLandmark.NOSE.value:
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]  # 鼻子关键点
        if nose.visibility > 0.3:  # 检查鼻子点是否可见
            head_x = int(nose.x * WIDTH)  # 头部X坐标（基于鼻子位置）
            head_y = int(nose.y * HEIGHT) - 40  # 头部Y坐标（在鼻子上方）

            # 绘制头部圆形
            pygame.draw.circle(screen, SKIN_COLOR, (head_x, head_y), HEAD_RADIUS)

            # 绘制头发 (二次元风格)
            pygame.draw.ellipse(screen, HAIR_COLOR,
                                (head_x - HEAD_RADIUS * 1.25,  # 头发X位置
                                 head_y - HEAD_RADIUS * 1.25,  # 头发Y位置
                                 HEAD_RADIUS * 2.5,  # 头发宽度
                                 HEAD_RADIUS * 1.5))  # 头发高度

            # 绘制眼睛
            eye_offset = 15  # 两眼间距
            # 左眼
            pygame.draw.circle(screen, (255, 255, 255),  # 眼白
                               (head_x - eye_offset, head_y - 5),
                               EYE_RADIUS)
            pygame.draw.circle(screen, (0, 0, 0),  # 瞳孔
                               (head_x - eye_offset, head_y - 5),
                               PUPIL_RADIUS)
            # 右眼
            pygame.draw.circle(screen, (255, 255, 255),
                               (head_x + eye_offset, head_y - 5),
                               EYE_RADIUS)
            pygame.draw.circle(screen, (0, 0, 0),
                               (head_x + eye_offset, head_y - 5),
                               PUPIL_RADIUS)

            # 绘制嘴巴（微笑）
            pygame.draw.arc(screen, (200, 100, 100),  # 嘴巴颜色
                            (head_x - 20, head_y + 10, 40, 20),  # 位置和大小
                            0, math.pi,  # 从0到π（180度）绘制弧形
                            2)  # 线条粗细


def process_frame(frame):
    """
    处理摄像头帧并检测姿态关键点
    参数:
        frame: 从摄像头捕获的图像帧
    返回:
        检测到的关键点列表（如果检测到），否则返回None
    """
    # MediaPipe需要RGB格式的图像，而OpenCV默认为BGR
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色空间
    image.flags.writeable = False  # 设置为不可写以提高性能

    # 使用MediaPipe进行姿态检测
    results = pose.process(image)

    # 如果检测到姿态，返回关键点
    if results.pose_landmarks:
        return results.pose_landmarks.landmark
    return None


# ======================== 主程序开始 ========================
# 打开摄像头
cap = cv2.VideoCapture(0)  # 参数0表示使用默认摄像头
# 设置摄像头分辨率（不保证所有摄像头都支持）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 高度

# 主循环标志
running = True
while running:
    # 处理Pygame事件（如退出）
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # 点击窗口关闭按钮
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  # 按下ESC键
                running = False

    # 从摄像头读取一帧
    ret, frame = cap.read()
    if not ret:  # 如果读取失败
        print("无法从摄像头获取帧")
        continue

    # 处理帧并获取人体关键点
    landmarks = process_frame(frame)

    # 清空屏幕（用背景色填充）
    screen.fill(BACKGROUND_COLOR)

    # 如果检测到关键点，绘制二次元人物
    if landmarks:
        draw_anime_character(screen, landmarks, CONNECTIONS)

    # 显示说明文本
    font = pygame.font.SysFont(None, 30)  # 使用默认字体，大小30
    text = font.render("摄像头动作捕捉驱动二次元人物 - 按ESC退出", True, (0, 0, 0))
    screen.blit(text, (20, 20))

    # 显示帧率
    fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, (0, 0, 0))
    screen.blit(fps_text, (20, HEIGHT - 40))

    # 更新显示
    pygame.display.flip()

    # 控制帧率（每秒30帧）
    clock.tick(30)

# ======================== 程序结束清理 ========================
# 释放摄像头资源
cap.release()
# 关闭Pygame
pygame.quit()
print("程序已退出")