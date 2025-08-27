import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

from game_manager import GameManager
from game_state import GameState
from skill_library import Skill, LevelUpSkill

# 각 보상 요소에 대한 가중치 설정 (하이퍼파라미터)
REWARD_CONFIG = {
    'event': 10.0,
    'badge': 50.0,
    'new_map': 5.0,
    'exploration': 0.1,
    'level_up': 2.0,
    'hp_penalty': -0.01,
}

MAX_EPISODE_STEPS = 131072

class PokemonGoldEnv(gym.Env):
    def __init__(self, rom_path: str, state_path: str = None, render_mode: str = None):
        super().__init__()
        
        self.metadata = {'render.modes': ['rgb_array'], 'render_fps': 4}
        self.render_mode = render_mode

        self.initial_state_path = state_path
        self.manager = GameManager(rom_path, state_path=state_path, headless=True)
        self.state_reader = GameState(self.manager.pyboy, rom_path=rom_path)
        
        self.action_space = spaces.Discrete(len(self.manager.action_map))
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(1, 144, 160), dtype=np.uint8),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32) 
        })
        self.current_skill: Skill = LevelUpSkill(target_level=251) # 기본 스킬
        self.main_task: str = "Become the Johto Champion"

        self.init_state()

    def set_attr(self, attr_name: str, value):
        """환경의 속성을 외부에서 설정하기 위한 메서드입니다."""
        setattr(self, attr_name, value)

    def init_state(self):
        """학습 에피소드에 필요한 상태 변수들을 초기화합니다."""
        self.current_state = {}
        self.reward_log = {} # 보상 디버깅용

        # 탐험 보상을 위한 변수
        self.seen_coords = {} # map_key -> set((x, y))
        
        # 중복 보상을 막기 위한 변수
        self.max_party_level_sum = 0
        self.max_badges = 0
        self.completed_events = set()

        self.step_count = 0 # <<< 에피소드 스텝 카운터

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.init_state()
        
        # <<< 수정: reset 시 항상 초기 .state 파일 또는 최고 .state 파일을 사용하도록 설정
        # main loop에서 self.manager.state_path를 변경해주는 방식 사용
        self.manager.reset() 
        self.current_state = self.state_reader.get_state_dict()

        self.max_party_level_sum = self.current_state['party_info']['party_level_sum']
        self.max_badges = self.current_state['player_info']['johto_badges_count']
        for event, status in self.current_state['event_statuses'].items():
            if status:
                self.completed_events.add(event)
        
        return self._get_observation(), self.current_state
    
    def render(self, mode='rgb_array'):
        """환경의 현재 화면을 numpy 배열로 반환합니다."""
        # GameManager를 통해 현재 화면 이미지를 가져옴
        return self.manager.get_screen_image()

    def _get_state_vector(self) -> np.ndarray:
        """ RAM에서 읽은 주요 정보들을 정규화하여 벡터로 만듭니다. """
        player_info = self.current_state.get('player_info', {})
        location_info = self.current_state.get('location', {})
        
        # 좌표와 맵 ID는 학습 안정성을 위해 -1 ~ 1 사이 값으로 정규화
        # 최대 맵 크기를 대략 255x255로 가정
        x_coord = (location_info.get('x_coord', 0) / 128.0) - 1.0
        y_coord = (location_info.get('y_coord', 0) / 128.0) - 1.0
        map_bank = (location_info.get('map_bank', 0) / 128.0) - 1.0
        map_id = (location_info.get('map_id', 0) / 128.0) - 1.0
        
        # 배지 개수는 0 ~ 1 사이 값으로 정규화 (최대 8개)
        badges = player_info.get('johto_badges_count', 0) / 8.0
        
        return np.array([x_coord, y_coord, map_bank, map_id, badges], dtype=np.float32)

    def _get_observation(self):
        """ ✨ [핵심 수정 2] 관측 데이터를 Dict 형태로 조합하여 반환합니다. """
        # VecFrameStack 래퍼는 Dict의 'image' 키에 자동으로 적용됩니다.
        # 따라서 우리는 채널이 1인 단일 이미지만 제공하면 됩니다.
        image_obs = np.transpose(self.manager.get_screen_image(), (2, 0, 1)) # HWC -> CHW
        state_vec = self._get_state_vector()
        
        return {"image": image_obs.astype(np.uint8), "state": state_vec}

    def _get_auxiliary_rewards(self, prev_state: dict) -> float:
        """
        항상 계산되는 보조적인 보상들을 계산합니다. (포켓몬 레드 프로젝트 아이디어 차용)
        """
        aux_reward = 0
        loc = self.current_state['location']
        map_key = f"{loc['map_bank']}_{loc['map_id']}"
        coords = (loc['x_coord'], loc['y_coord'])
        if map_key not in self.seen_coords:
            self.seen_coords[map_key] = set()
            aux_reward += 5.0
        if coords not in self.seen_coords[map_key]:
            self.seen_coords[map_key].add(coords)
            aux_reward += 0.1
        hp_lost = prev_state['party_info']['party_hp_sum'] - self.current_state['party_info']['party_hp_sum']
        if hp_lost > 0:
            aux_reward -= hp_lost * 0.01
        if not prev_state.get('is_in_menu') and self.current_state.get('is_in_menu'):
            aux_reward -= 0.5  # 메뉴를 '새로 열 때' 큰 페널티 (START 버튼 억제)
        if prev_state.get('is_in_menu') and self.current_state.get('is_in_menu'):
            aux_reward -= 0.1  # 메뉴에 '머무를 때' 지속 페널티 (메뉴 내 불필요한 액션 억제)
        return aux_reward

    #def _calculate_navigation_reward(self, prev_state: dict, current_state: dict) -> float:
    #    """탐색(Navigation) 중에 사용될 보상 함수입니다."""
    #    # LLM이 선택한 현재 스킬에 따라 보상을 계산
    #    return self.current_skill.get_reward(prev_state, current_state)

    def _calculate_battle_reward(self, prev_state: dict, current_state: dict) -> float:
        """전투 상황에 맞는 보상을 정교하게 계산합니다."""
        reward = 0
        prev_battle_info = prev_state.get('battle_info', {})
        curr_battle_info = current_state.get('battle_info', {})
        if 'enemy_hp' in prev_battle_info and 'enemy_hp' in curr_battle_info:
            damage_dealt = prev_battle_info['enemy_hp'] - curr_battle_info['enemy_hp']
            if damage_dealt > 0:
                reward += damage_dealt * 0.5
        hp_lost = prev_state['party_info']['party_hp_sum'] - current_state['party_info']['party_hp_sum']
        if hp_lost > 0:
            reward -= hp_lost * 0.1
        if prev_state.get('is_in_battle') and not current_state.get('is_in_battle'):
            if prev_battle_info.get('enemy_hp', 0) == 0:
                 reward += 100.0
            else:
                 reward -= 10.0
        return reward
    
    def step(self, action: int):
        self.step_count += 1 # <<< 스텝 수 증가
        prev_state = self.current_state
        
        self.manager.step(action)
        self.current_state = self.state_reader.get_state_dict()
        
        obs = self._get_observation()
        
        if self.current_state['is_in_battle']:
            main_reward = self._calculate_battle_reward(prev_state, self.current_state)
        else:
            main_reward = self.current_skill.get_reward(prev_state, self.current_state)
            
        aux_reward = self._get_auxiliary_rewards(prev_state)

        reward = main_reward + aux_reward

        if prev_state['party_info']['party_hp_sum'] > 0 and self.current_state['party_info']['party_hp_sum'] == 0:
            reward -= 50.0

        # 종료 조건
        # 1. 파티가 전멸했을 때 - deleted
        terminated = False
        # 2. 최대 스텝 수를 초과했을 때 (Truncated)
        truncated = self.step_count >= MAX_EPISODE_STEPS
        
        info = self.current_state
        
        return obs, reward, terminated, truncated, info

    def save_state(self, path: str):
        """GameManager를 통해 현재 게임 상태를 저장합니다."""
        self.manager.save_state(path)

    def _get_rewards_dict(self, prev_state: dict) -> dict[str, float]:
        """PokeRL 스타일로 보상 요소들을 개별적으로 계산하여 딕셔너리로 반환합니다."""
        rewards = {}
        
        # 1. 이벤트 보상
        event_reward = 0
        for event, status in self.current_state['event_statuses'].items():
            if status and event not in self.completed_events:
                event_reward += 1
                self.completed_events.add(event)
        rewards['event'] = event_reward * REWARD_CONFIG['event']

        # 2. 배지 보상
        badge_reward = self.current_state['player_info']['johto_badges_count'] - self.max_badges
        if badge_reward > 0:
            self.max_badges = self.current_state['player_info']['johto_badges_count']
        rewards['badge'] = badge_reward * REWARD_CONFIG['badge']

        # 3. 레벨 보상
        level_reward = self.current_state['party_level_sum'] - self.max_party_level_sum
        if level_reward > 0:
            self.max_party_level_sum = self.current_state['party_level_sum']
        rewards['level_up'] = level_reward * REWARD_CONFIG['level_up']

        # 4. 탐험 보상
        loc = self.current_state['location']
        map_key = f"{loc['map_bank']}_{loc['map_id']}"
        coords = (loc['x_coord'], loc['y_coord'])
        
        if map_key not in self.seen_coords:
            self.seen_coords[map_key] = set()
            rewards['new_map'] = REWARD_CONFIG['new_map']
        
        if coords not in self.seen_coords[map_key]:
            self.seen_coords[map_key].add(coords)
            rewards['exploration'] = REWARD_CONFIG['exploration']

        # 5. HP 감소 페널티
        hp_penalty = self.current_state['party_info']['party_hp_sum'] - prev_state['party_info']['party_hp_sum']
        if hp_penalty < 0:
            rewards['hp_penalty'] = hp_penalty * REWARD_CONFIG['hp_penalty']

        # 0이 아닌 보상만 필터링하여 반환
        return {k: v for k, v in rewards.items() if v != 0}