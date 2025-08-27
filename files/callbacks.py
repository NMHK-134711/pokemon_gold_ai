import pandas as pd
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
import numpy as np
import torch as th

class EpisodeLogCallback(BaseCallback):
    def __init__(self, log_path: str, verbose=0):
        super(EpisodeLogCallback, self).__init__(verbose)
        self.log_path = log_path
        self.episode_num = 0
        # 로그 파일이 없으면 헤더와 함께 새로 생성
        if not os.path.exists(self.log_path):
            pd.DataFrame(columns=[
                'episode', 'total_steps', 'episode_reward', 
                'map_bank', 'map_id', 'pos_x', 'pos_y', 
                'party_level_sum', 'badges', 'money', 'events_completed'
            ]).to_csv(self.log_path, index=False)

    def _on_step(self) -> bool:
        # VecEnv에서는 dones가 여러 환경의 종료 여부를 담은 배열
        for i, done in enumerate(self.locals['dones']):
            if done:
                self.episode_num += 1
                
                # 종료된 환경의 정보(info) 가져오기
                info = self.locals['infos'][i]
                
                # 로그 데이터 정리
                log_data = {
                    'episode': self.episode_num,
                    'total_steps': self.num_timesteps,
                    'episode_reward': info.get('episode', {}).get('r', 0),
                    'map_bank': info.get('location', {}).get('map_bank', 0),
                    'map_id': info.get('location', {}).get('map_id', 0),
                    'pos_x': info.get('location', {}).get('x_coord', 0),
                    'pos_y': info.get('location', {}).get('y_coord', 0),
                    'party_level_sum': info.get('party_info', {}).get('party_level_sum', 0),
                    'badges': info.get('player_info', {}).get('johto_badges_count', 0),
                    'money': info.get('player_info', {}).get('money', 0),
                    'events_completed': sum(info.get('event_statuses', {}).values())
                }
                
                # 로그를 CSV 파일에 추가
                df = pd.DataFrame([log_data])
                df.to_csv(self.log_path, mode='a', header=False, index=False)
                
                if self.verbose > 0:
                    print(f"Agent {i} finished episode {self.episode_num}. Log saved to {self.log_path}")

        return True

class BestAgentCallback(BaseCallback):
    """
    최고 성과를 내는 에이전트의 모델과 상태를 저장하는 콜백.
    """
    def __init__(self, nav_model_path: str, battle_model_path: str, best_state_path: str, verbose=0):
        super(BestAgentCallback, self).__init__(verbose)
        self.nav_model_path = nav_model_path
        self.battle_model_path = battle_model_path
        self.best_state_path = best_state_path
        
        # 최고 점수 추적 (요청된 우선순위 기준)
        self.best_score = {
            "events_completed": -1,
            "episode_reward": -np.inf,
            "badges": -1,
            "party_level_sum": -1,
            "money": -1
        }

    def _is_new_score_better(self, new_score: dict) -> bool:
        """새로운 점수가 기존 최고 점수보다 나은지 우선순위에 따라 확인합니다."""
        priority = ["events_completed", "episode_reward", "badges", "party_level_sum", "money"]
        for key in priority:
            if new_score[key] > self.best_score[key]:
                return True
            if new_score[key] < self.best_score[key]:
                return False
        return False # 모든 점수가 동일한 경우

    def _on_step(self) -> bool:
        # 여러 환경(에이전트)을 순회하며 에피소드 종료 여부 확인
        for i, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][i]
                
                current_score = {
                    "events_completed": sum(info.get('event_statuses', {}).values()),
                    "episode_reward": info.get('episode', {}).get('r', 0),
                    "badges": info.get('player_info', {}).get('johto_badges_count', 0),
                    "party_level_sum": info.get('party_info', {}).get('party_level_sum', 0),
                    "money": info.get('player_info', {}).get('money', 0)
                }

                if self._is_new_score_better(current_score):
                    self.best_score = current_score
                    print("\n" + "="*50)
                    print(f"🎉 새로운 최고 기록 달성! (by Agent {i})")
                    print(f"   - 이벤트: {self.best_score['events_completed']}, 보상: {self.best_score['episode_reward']:.2f}, "
                          f"배지: {self.best_score['badges']}, 레벨 합: {self.best_score['party_level_sum']}")
                    print("   - 최고 모델과 게임 상태를 저장합니다...")
                    print("="*50 + "\n")

                    # 현재 학습 중인 모델을 '최고 모델'로 저장
                    # self.model은 RecurrentPPO 인스턴스를 가리킴
                    # train_hierarchical.py에서 어떤 모델이 learn()을 호출했는지에 따라 저장됨
                    if info.get('is_in_battle'):
                        self.model.save(self.battle_model_path)
                    else:
                        self.model.save(self.nav_model_path)
                    
                    # 해당 에이전트의 게임 상태를 '최고 상태'로 저장
                    self.training_env.env_method('save_state', self.best_state_path, indices=[i])

        return True

class ImageLogCallback(BaseCallback):
    def __init__(self, frame_interval: int = 1024, verbose=0):
        super(ImageLogCallback, self).__init__(verbose)
        self.frame_interval = frame_interval

    def _on_step(self) -> bool:
        if self.n_calls % self.frame_interval == 0:
            
            obs_dict = self.model._last_obs
          
            image_batch = obs_dict["image"]
       
            latest_frame_batch = image_batch[:, -1:, :, :]
            
            # 이제 1채널 이미지를 TensorBoard에 전달합니다.
            tb_image = Image(latest_frame_batch.astype(np.uint8), 'NCHW')

            self.logger.record(
                "agent_views/all_agents", 
                tb_image, 
                exclude=("stdout", "log", "csv")
            )
        return True