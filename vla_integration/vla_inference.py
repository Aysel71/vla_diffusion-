"""
Запуск SmolVLA через LeRobot с актуальным API (через pre/post‑processors).
"""

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.control_utils import predict_action
from PIL import Image
import numpy as np

# ============================================
# 1. Загрузка модели
# ============================================

print("Загружаем SmolVLA через LeRobot...")

# Загружаем предобученную модель
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

# Используем CUDA (если доступна) или CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = policy.to(device)
policy.eval()

print(f"Модель загружена на {device}")

# Создаём pre/post‑processors для SmolVLA
preprocessor, postprocessor = make_smolvla_pre_post_processors(policy.config, dataset_stats=None)

# ============================================
# 2. Подготовка наблюдения
# ============================================

# Создаём тестовое изображение
test_image = Image.new("RGB", (512, 512), color="lightgray")
# Или загрузи своё: test_image = Image.open("robot_view.jpg")

# Инструкция
instruction = "Pick up the red cube and place it in the box"

# Состояние робота (начальная позиция)
robot_state = np.array([0.0, -0.5, 0.5, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)

# Наблюдение в формате numpy, как ожидает prepare_observation_for_inference
raw_observation = {
    "observation.images.camera1": np.array(test_image),
    "observation.state": robot_state,
}

# ============================================
# 3. Запуск инференса через стандартный pipeline LeRobot
# ============================================

print(f"\nИнструкция: {instruction}")
print("Генерируем действия...")

action = predict_action(
    observation=raw_observation,
    policy=policy,
    device=device,
    preprocessor=preprocessor,
    postprocessor=postprocessor,
    use_amp=False,
    task=instruction,
    robot_type=None,
)

print(f"\nПредсказанные действия:")
print(action)