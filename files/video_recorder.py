# video_recorder.py

import os
import numpy as np
import gymnasium as gym
from PIL import Image

class VideoRecorderWrapper(gym.Wrapper):
    def __init__(self, env, save_dir: str, frame_interval: int = 4, env_rank: int = 0):
        super().__init__(env)
        
        # ❗️ 에이전트별 고유 폴더 경로 설정
        self.save_dir = os.path.join(save_dir, f"agent_{env_rank}")
        
        self.frame_interval = frame_interval
        self.step_count = 0
        self.episode_count = 0
        
        os.makedirs(self.save_dir, exist_ok=True)

    def reset(self, **kwargs):
        # reset이 호출될 때마다 에피소드 카운트를 1 증가시킵니다.
        self.episode_count += 1
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.step_count % self.frame_interval == 0:
            # ❗️ 파일 이름에 에피소드 번호를 포함하여 저장
            filename = f"ep{self.episode_count:03d}_step{self.step_count:06d}.png"
            frame_path = os.path.join(self.save_dir, filename)
            
            # 이미지 데이터 처리
            img_array = obs[0] if isinstance(obs, tuple) else obs
            if img_array.shape[-1] == 1:
                img_array = img_array.squeeze(-1)

            img = Image.fromarray(img_array)
            img.save(frame_path)
            
        self.step_count += 1
        
        return obs, reward, terminated, truncated, info