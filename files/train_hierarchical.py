import os
import json

from sb3_contrib import RecurrentPPO 
from stable_baselines3.common.vec_env import SubprocVecEnv
from pokemon_env import PokemonGoldEnv
from llm_planner import LLMPlanner
from skill_library import AVAILABLE_SKILLS, HealPartySkill
from task_manager import TaskManager
from callbacks import EpisodeLogCallback, BestAgentCallback, ImageLogCallback
from custom_policy import CombinedExtractor
from concurrent.futures import ThreadPoolExecutor
from custom_wrappers import VecDictFrameStack

# --- 하이퍼파라미터 ---
ROM_PATH = "PokemonGold.gbc"
INITIAL_STATE_PATH = "initial_gold.state"
PLAN_PATH = "pokemon_gold_plan.json"
TOTAL_TRAINING_STEPS = 1_000_000
STEPS_PER_SEGMENT = 8192
MODEL_SAVE_PATH = 'trained_models'
LOG_DIR = 'logs'
NUM_ENVS = 8
SYNC_INTERVAL = 10 

POKEMON_CENTERS = [
    {'name': 'new bark town', 'map_bank': 24, 'map_id': 5, 'x': 2, 'y': 2},
    {'name': '무궁시티 포켓몬 센터', 'map_bank': 24, 'map_id': 7, 'x': 10, 'y': 2},
    {'name': '도라지시티 포켓몬 센터', 'map_bank': 3, 'map_id': 1, 'x': 12, 'y': 2},
    {'name': '고동마을 포켓몬 센터', 'map_bank': 4, 'map_id': 0, 'x': 8, 'y': 6},
    {'name': '금빛시티 포켓몬 센터', 'map_bank': 5, 'map_id': 0, 'x': 14, 'y': 8},
]

def needs_healing(game_state: dict) -> bool:
    party = game_state.get('party_info', {}).get('pokemon', [])
    if not party: return False
    return any(
        p.get('current_hp', 1) == 0 or 
        (p.get('max_hp', 0) > 0 and (p.get('current_hp', 1) / p.get('max_hp', 1)) <= 0.3)
        for p in party
    )

def get_heal_skill(game_state: dict) -> HealPartySkill:
    loc = game_state['location']
    for center in POKEMON_CENTERS:
        if center['map_bank'] == loc['map_bank'] and center['map_id'] == loc['map_id']:
            return HealPartySkill(center)
    return HealPartySkill(POKEMON_CENTERS[0])

def make_env(rank: int, state_path: str):
    def _init():
        env = PokemonGoldEnv(
            rom_path=ROM_PATH, 
            state_path=state_path, 
            render_mode='rgb_array'
        )
        return env
    return _init

def main():
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    executor = ThreadPoolExecutor(max_workers=1)
    llm_future = None

    best_nav_model_path = os.path.join(MODEL_SAVE_PATH, "best_nav_model.zip")
    best_battle_model_path = os.path.join(MODEL_SAVE_PATH, "best_battle_model.zip")
    best_state_path = "best_agent.state"

    log_callback = EpisodeLogCallback(log_path="episode_log.csv")
    best_agent_callback = BestAgentCallback(
        nav_model_path=best_nav_model_path,
        battle_model_path=best_battle_model_path,
        best_state_path=best_state_path
    )
    image_callback = ImageLogCallback(frame_interval=1024)

    vec_env = SubprocVecEnv([make_env(i, INITIAL_STATE_PATH) for i in range(NUM_ENVS)])
    vec_env = VecDictFrameStack(vec_env, n_stack=4, dict_obs_key="image")

    planner = LLMPlanner()
    task_manager = TaskManager(plan_path=PLAN_PATH)

    nav_model_to_load = best_nav_model_path if os.path.exists(best_nav_model_path) else os.path.join(MODEL_SAVE_PATH, "nav_ppo_model.zip")
    battle_model_to_load = best_battle_model_path if os.path.exists(best_battle_model_path) else os.path.join(MODEL_SAVE_PATH, "battle_ppo_model.zip")

    policy_kwargs = {
        "features_extractor_class": CombinedExtractor,
    }

    if os.path.exists(nav_model_to_load):
        print(f"탐색 모델을 로드합니다: {nav_model_to_load}")
        nav_model = RecurrentPPO.load(nav_model_to_load, env=vec_env, policy_kwargs=policy_kwargs, tensorboard_log=LOG_DIR)
    else:
        print("새로운 탐색 모델을 생성합니다.")
        # ✨ [수정 3] 'CnnLstmPolicy'를 사용하고, policy_kwargs를 전달합니다.
        nav_model = RecurrentPPO(
            'CnnLstmPolicy', 
            vec_env, 
            policy_kwargs=policy_kwargs, 
            verbose=1, 
            tensorboard_log=LOG_DIR, 
            n_steps=STEPS_PER_SEGMENT
        )

    if os.path.exists(battle_model_to_load):
        print(f"배틀 모델을 로드합니다: {battle_model_to_load}")
        battle_model = RecurrentPPO.load(battle_model_to_load, env=vec_env, policy_kwargs=policy_kwargs, tensorboard_log=LOG_DIR)
    else:
        print("새로운 배틀 모델을 생성합니다.")
        # ✨ [수정 4] 배틀 모델에도 동일하게 적용합니다.
        battle_model = RecurrentPPO(
            'CnnLstmPolicy', 
            vec_env, 
            policy_kwargs=policy_kwargs, 
            verbose=1, 
            tensorboard_log=LOG_DIR, 
            n_steps=STEPS_PER_SEGMENT
        )
    vec_env.reset()
    initial_info = vec_env.get_attr('current_state')[0]
    task_manager.sync_with_initial_state(initial_info)
    
    total_steps = 0
    segment_count = 0
    while total_steps < TOTAL_TRAINING_STEPS:
        segment_count += 1
        all_current_infos = vec_env.get_attr('current_state')
        
        if segment_count % SYNC_INTERVAL == 0 and os.path.exists(best_state_path):
            print("\n" + "#"*60)
            print(f"🔄 동기화 시점 도달. 모든 에이전트를 최고 상태({best_state_path})에서 재시작합니다.")
            print("#"*60 + "\n")
            vec_env.set_attr('manager.state_path', best_state_path)
            vec_env.reset()
            # 리셋 후에도 상태를 다시 가져옵니다.
            all_current_infos = vec_env.get_attr('current_state')
            task_manager.sync_with_initial_state(all_current_infos[0])

        # 첫 번째 에이전트의 상태를 기준으로 배틀 여부를 판단합니다.
        if all_current_infos[0]['is_in_battle']:
            print("\n--- 배틀 모드 ---")
            current_model = battle_model
            callback_to_use = [log_callback, best_agent_callback, image_callback]
        else:
            # --- 탐색 모드 ---
            print("\n--- 탐색 모드 ---")
            current_model = nav_model
            
            # 1. 이전 LLM 작업이 완료되었는지 확인하고 결과 적용
            if llm_future and llm_future.done():
                chosen_skills = llm_future.result()
                print("✅ 백그라운드 LLM 작업 완료! 새로운 스킬을 적용합니다.")
                for i, skill in enumerate(chosen_skills):
                    vec_env.env_method('set_attr', 'current_skill', skill, indices=[i])
                    print(f"    -> Agent {i} 하위 목표: [{skill.description}]")
                llm_future = None # 작업 완료 후 초기화

            # 2. 새로운 LLM 작업이 필요한지 확인하고 백그라운드로 실행
            if llm_future is None:
                while task_manager.is_current_task_completed(all_current_infos[0]):
                    task_manager.advance_to_next_task()
                
                main_task = task_manager.get_current_task_description()
                vec_env.set_attr('main_task', main_task)
                print(f"[주요 목표: {main_task}]")
                print("  - 🧠 다음 하위 목표 결정을 LLM에 비동기로 요청합니다...")
                
                # LLM 호출을 백그라운드 스레드에서 실행
                llm_future = executor.submit(
                    planner.choose_next_skill_batch, 
                    all_current_infos, 
                    main_task, 
                    AVAILABLE_SKILLS
                )

        # 3. LLM 호출과 상관없이, 현재 스킬로 학습을 즉시 진행
        current_model.learn(
            total_timesteps=STEPS_PER_SEGMENT, 
            reset_num_timesteps=False, 
            tb_log_name="RecurrentPPO",
            callback=[log_callback, best_agent_callback, image_callback],
        )
        total_steps += STEPS_PER_SEGMENT
        
        print(f"총 진행 스텝: {total_steps}/{TOTAL_TRAINING_STEPS} (세그먼트: {segment_count})")

        if total_steps % (STEPS_PER_SEGMENT * 2) == 0:
            print("모델을 주기적으로 저장합니다...")
            nav_model.save(os.path.join(MODEL_SAVE_PATH, "nav_ppo_model.zip"))
            battle_model.save(os.path.join(MODEL_SAVE_PATH, "battle_ppo_model.zip"))

    executor.shutdown()
    print("***** 전체 학습 종료 *****")
    nav_model.save(os.path.join(MODEL_SAVE_PATH, "nav_ppo_final.zip"))
    battle_model.save(os.path.join(MODEL_SAVE_PATH, "battle_ppo_final.zip"))
    vec_env.close()

if __name__ == "__main__":
    if not os.path.exists(PLAN_PATH):
        plan_data = {
          "goal": "Become the Johto Champion",
          "tasks": [
            "Start game and choose Cyndaquil as starter",
            "Obtain Pokédex and Poké Balls",
            "Defeat Gym 1 - Falkner (Flying, Violet City)"
          ]
        }
        with open(PLAN_PATH, 'w', encoding='utf-8') as f:
            json.dump(plan_data, f, indent=2, ensure_ascii=False)
    
    main()
