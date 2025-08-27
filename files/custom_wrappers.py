import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from collections import deque

class VecDictFrameStack(VecEnvWrapper):
    def __init__(self, venv: VecEnv, n_stack: int, dict_obs_key: str):
        self.venv = venv
        self.n_stack = n_stack
        self.dict_obs_key = dict_obs_key
        self.stacked_obs = None

        # 관측 공간(observation space)을 수정합니다.
        # 기존 딕셔너리 공간을 복사한 뒤, 스택을 적용할 키의 공간만 모양(shape)을 바꿔줍니다.
        wrapped_obs_space = venv.observation_space
        self.original_image_space = wrapped_obs_space.spaces[self.dict_obs_key]
        
        # 채널(channel) 차원을 맨 앞으로 가정 (CHW 포맷)
        low = np.repeat(self.original_image_space.low, self.n_stack, axis=0)
        high = np.repeat(self.original_image_space.high, self.n_stack, axis=0)
        
        # 새로운 이미지 공간 생성
        stacked_image_space = spaces.Box(
            low=low, high=high, dtype=self.original_image_space.dtype
        )
        
        # 전체 관측 공간 업데이트
        new_spaces = {k: v for k, v in wrapped_obs_space.spaces.items()}
        new_spaces[self.dict_obs_key] = stacked_image_space
        
        super().__init__(venv, observation_space=spaces.Dict(new_spaces))

        # 환경별 프레임 버퍼를 초기화합니다.
        self.buffers = [deque([], maxlen=self.n_stack) for _ in range(self.num_envs)]

    def _get_stacked_obs(self):
        # 버퍼에서 프레임들을 가져와 하나의 numpy 배열로 합칩니다.
        # 각 버퍼를 리스트로 변환한 뒤 numpy 배열로 만듭니다.
        stacked_images = np.array([list(buf) for buf in self.buffers])
        
        # (환경 수, 스택 수, 채널, 높이, 너비) -> (환경 수, 스택*채널, 높이, 너비)
        # 예: (4, 4, 1, 144, 160) -> (4, 4, 144, 160)
        b, s, c, h, w = stacked_images.shape
        return stacked_images.reshape(b, s * c, h, w)

    def _process_obs(self, obs):
        """딕셔너리 관측을 받아 스택된 버전으로 교체합니다."""
        stacked_images = self._get_stacked_obs()
        # 원본 딕셔너리에서 이미지 부분만 스택된 이미지로 교체합니다.
        obs[self.dict_obs_key] = stacked_images
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        
        for i in range(self.num_envs):
            # 에피소드가 끝나면(done), 버퍼를 초기 프레임으로 다시 채웁니다.
            if dones[i]:
                self.buffers[i].clear()
                # VecEnv는 자동으로 리셋하므로, infos에서 'terminal_observation'을 가져옵니다.
                terminal_obs_image = infos[i]['terminal_observation'][self.dict_obs_key]
                for _ in range(self.n_stack):
                    self.buffers[i].append(terminal_obs_image)
            else:
                self.buffers[i].append(obs[self.dict_obs_key][i])
        
        return self._process_obs(obs), rewards, dones, infos

    def reset(self):
        obs = self.venv.reset()
        # 리셋 시, 모든 버퍼를 첫 번째 프레임으로 가득 채웁니다.
        initial_images = obs[self.dict_obs_key]
        for i in range(self.num_envs):
            self.buffers[i].clear()
            for _ in range(self.n_stack):
                self.buffers[i].append(initial_images[i])
        
        return self._process_obs(obs)

    def close(self):
        self.venv.close()