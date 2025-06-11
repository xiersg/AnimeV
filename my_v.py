import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import os
import sys


class AnimeCharacterDriver:
    def __init__(self, resource_dir, camera_index=0, window_size=(1000, 700)):

        self.resource_dir = resource_dir        # 体块图片位置
        self.camera_index = camera_index        # 摄像头编号
        self.width, self.height = window_size   # 可视化窗口长宽

        # 初始化MediaPipe姿态检测模型
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2
        )

        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("骨骼绑定二次元人物驱动 - 按ESC退出")
        self.clock = pygame.time.Clock()

        # 加载角色部件
        self.character_parts = self.load_character_parts()

        # 定义部件绑定关系
        self.PART_BINDINGS = {
            "head": (self.mp_pose.PoseLandmark.NOSE, None),
            "body": (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            "left_upper_arm": (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            "left_lower_arm": (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            "left_hand": (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.LEFT_PINKY),
            "right_upper_arm": (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            "right_lower_arm": (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            "right_hand": (self.mp_pose.PoseLandmark.RIGHT_WRIST, self.mp_pose.PoseLandmark.RIGHT_PINKY),
            "left_upper_leg": (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            "left_lower_leg": (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            "left_foot": (self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.LEFT_HEEL),
            "right_upper_leg": (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            "right_lower_leg": (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            "right_foot": (self.mp_pose.PoseLandmark.RIGHT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_HEEL)
        }

        # 渲染顺序
        self.RENDER_ORDER = [
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
        self.BACKGROUND_COLOR = (240, 248, 255)

        # 人物位置偏移
        self.character_offset_x = self.width // 2
        self.character_offset_y = self.height // 2

        # 打开摄像头
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 加载字体
        try:
            self.font = pygame.font.SysFont("microsoftyahei", 24)
        except:
            self.font = pygame.font.SysFont(None, 24)


    def load_character_parts(self):
        """加载角色部件资源"""
        BODY_PARTS = {
            "head": {"file": "head.png", "anchor": (0.5, 0.8)},
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

        character_parts = {}
        for part, info in BODY_PARTS.items():
            try:
                img_path = os.path.join(self.resource_dir, info["file"])
                img = pygame.image.load(img_path)
                img = img.convert_alpha()
                character_parts[part] = {
                    "image": img,
                    "anchor": info["anchor"],
                    "rect": img.get_rect()
                }
                print(f"加载部件: {part} 尺寸: {img.get_size()}")
            except Exception as e:
                print(f"警告: 无法加载部件 {info['file']}: {e}, 将使用占位图形")
                surf = pygame.Surface((50, 50), pygame.SRCALPHA)
                pygame.draw.rect(surf, (255, 0, 0, 128), (0, 0, 50, 50))
                character_parts[part] = {
                    "image": surf,
                    "anchor": (0.5, 0.5),
                    "rect": surf.get_rect()
                }
        return character_parts

# ----------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def calculate_rotation(start_point, end_point):
        """计算两点之间的旋转角度"""
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        angle = math.degrees(math.atan2(-dy, dx))
        return angle

    def transform_part(self, part, landmarks, binding, scale=1.0):
        """变换身体部件"""
        # 获取绑定的关键点
        start_idx = binding[0].value
        end_idx = binding[1].value if binding[1] else None

        # 获取起始点位置
        start_point = landmarks[start_idx]
        start_x = int(start_point["x"] * self.width)
        start_y = int(start_point["y"] * self.height)

        # 计算旋转角度
        angle = 0
        if end_idx is not None and end_idx < len(landmarks):
            end_point = landmarks[end_idx]
            end_x = int(end_point["x"] * self.width)
            end_y = int(end_point["y"] * self.height)
            angle = self.calculate_rotation((start_x, start_y), (end_x, end_y))

        # 获取原始图像并缩放
        original_image = part["image"]
        if scale != 1.0:
            new_size = (int(original_image.get_width() * scale),
                        int(original_image.get_height() * scale))
            scaled_image = pygame.transform.scale(original_image, new_size)
        else:
            scaled_image = original_image

        # 旋转图像
        rotated_image = pygame.transform.rotate(scaled_image, -angle)

        # 计算锚点偏移
        anchor_x = int(part["anchor"][0] * scaled_image.get_width())
        anchor_y = int(part["anchor"][1] * scaled_image.get_height())

        # 计算旋转后的位置
        rotated_rect = rotated_image.get_rect()
        pos_x = start_x - rotated_rect.width // 2
        pos_y = start_y - rotated_rect.height // 2

        return rotated_image, (pos_x, pos_y)


    def draw_character(self, landmarks):
        """绘制骨骼绑定的人物"""
        if not landmarks:
            return

        # 创建偏移关键点
        offset_landmarks = []
        for lm in landmarks:
            offset_landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "visibility": lm.visibility
            })

        # 调整位置
        for lm in offset_landmarks:
            lm["x"] = lm["x"] * 0.7 + 0.15
            lm["y"] = lm["y"] * 0.7 + 0.15

        # 按顺序绘制部件
        for part_name in self.RENDER_ORDER:
            if part_name in self.character_parts and part_name in self.PART_BINDINGS:
                part = self.character_parts[part_name]
                binding = self.PART_BINDINGS[part_name]
                img, pos = self.transform_part(part, offset_landmarks, binding)
                self.screen.blit(img, pos)

    def process_frame(self, frame):
        """处理摄像头帧并检测姿态关键点"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        return results.pose_landmarks.landmark if results.pose_landmarks else None


    def handle_events(self):
        """处理Pygame事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_UP:
                    self.character_offset_y -= 10
                elif event.key == pygame.K_DOWN:
                    self.character_offset_y += 10
                elif event.key == pygame.K_LEFT:
                    self.character_offset_x -= 10
                elif event.key == pygame.K_RIGHT:
                    self.character_offset_x += 10
        return True

    def run(self):
        """主运行循环"""
        running = True
        while running:
            # 处理事件
            running = self.handle_events()

            # 读取摄像头帧
            ret, frame = self.cap.read()
            if not ret:
                print("无法从摄像头获取帧")
                continue

            # 水平镜像翻转
            frame = cv2.flip(frame, 1)

            # 处理帧并获取关键点
            landmarks = self.process_frame(frame)

            # 清空屏幕
            self.screen.fill(self.BACKGROUND_COLOR)

            # 绘制人物
            if landmarks:
                self.draw_character(landmarks)

            # 绘制UI元素
            title = self.font.render("骨骼绑定二次元人物驱动系统", True, (0, 0, 0))
            help_text = self.font.render("方向键移动人物位置 | ESC退出", True, (100, 100, 100))
            fps_text = self.font.render(f"帧率: {int(self.clock.get_fps())} FPS", True, (0, 0, 0))

            self.screen.blit(title, (20, 20))
            self.screen.blit(help_text, (20, self.height - 40))
            self.screen.blit(fps_text, (self.width - 150, self.height - 40))

            # 更新显示
            pygame.display.flip()
            self.clock.tick(30)

        # 清理资源
        self.cap.release()
        pygame.quit()


def main():
    # 获取资源路径
    # 创建并运行系统
    resource_dir = "D:\\AnimeV\\processed_character_parts\\character_parts"
    driver = AnimeCharacterDriver(
        resource_dir=resource_dir,
        camera_index=0,  # 默认摄像头
        window_size=(1200, 800)  # 自定义窗口尺寸
    )
    driver.run()
    # 创建并运行驱动系统


if __name__ == "__main__":
    main()