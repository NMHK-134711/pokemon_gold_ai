# task_manager.py
import json

# ==============================================================================
# DataCrystal 및 Bulbapedia 기반 상수 정의
# ==============================================================================
# 포켓몬 종족 ID (Species ID)
CYNDAQUIL = 155
HOOTHOOT = 163
MAREEP = 179
SLOWPOKE = 79
NIDORAN_M = 32
HO_OH = 250

# 주요 아이템 ID (Key Items, HMs, TMs)
# (주의: 실제 게임 내 ID와 다를 수 있으므로 DataCrystal에서 정확한 값 확인 필요)
MYSTERY_EGG_ITEM = 0xC5 # 이벤트 아이템, 인벤토리 확인 어려움
POKEDEX_ITEM = 0x00 # 가상 ID, 이벤트 플래그로 확인
MAP_CARD_ITEM = 0x37 # 포켓기어 카드
HM01_CUT = 0xBE
HM03_SURF = 0xC0
HM04_STRENGTH = 0xC1
HM05_FLASH = 0xB9
HM06_WHIRLPOOL = 0xC3
HM07_WATERFALL = 0xC4
SQUIRTBOTTLE = 0x4E
RAINBOW_WING = 0x9D
MOON_STONE = 0x0A

# 기술 ID (Move ID)
MUD_SLAP = 200
SHADOW_BALL = 247
STRENGTH = 70
CUT = 15
# ... 기타 필요한 기술 ID

# ==============================================================================
class TaskManager:
    def __init__(self, plan_path: str):
        with open(plan_path, 'r', encoding='utf-8') as f:
            self.plan = json.load(f)
        self.current_task_index = 0
        # helper method를 위한 game_state 저장 변수
        self.game_state = {}

    def _update_internal_state(self, game_state: dict):
        """내부 game_state를 최신으로 업데이트"""
        self.game_state = game_state

    # --- 검증 헬퍼 함수 ---
    def _has_pokemon(self, species_id: int) -> bool:
        party = self.game_state.get('party_info', {}).get('pokemon', [])
        return any(p['species_id'] == species_id for p in party)

    def _check_badges(self, count: int) -> bool:
        return self.game_state.get('player_info', {}).get('johto_badges_count', 0) >= count
    
    def _check_event_flag(self, event_name: str) -> bool:
        return self.game_state.get('event_statuses', {}).get(event_name, False)

    # _has_item, _pokemon_has_move 등은 GameState에 인벤토리/기술 읽기 기능 추가 후 구현 가능

    def get_current_task_description(self) -> str:
        """현재 수행해야 할 목표의 설명을 반환합니다."""
        if self.current_task_index < len(self.plan['tasks']):
            return self.plan['tasks'][self.current_task_index]
        return "모든 목표를 달성했습니다!"

    def is_current_task_completed(self, game_state: dict) -> bool:
        """json 계획의 각 태스크를 게임 상태와 대조하여 완료 여부를 상세히 확인합니다."""
        self._update_internal_state(game_state)
        if self.current_task_index >= len(self.plan['tasks']):
            return True

        task = self.get_current_task_description()

        # 각 태스크에 대한 상세한 완료 조건
        if "Start game" in task and "choose Cyndaquil" in task:
            return self._has_pokemon(CYNDAQUIL) and self._check_event_flag('starter_received')
        if "Get Map Card" in task:
            return self._check_event_flag('guide_gent_map')
        if "receive Mystery Egg and Pokédex" in task:
            return self._check_event_flag('got_pokedex')
        if "give Mystery Egg to Professor Elm" in task:
            # 알을 주고 나면 라이벌 이름을 짓고, 그 후 포켓몬을 받았다는 플래그가 뜸
            return self._check_event_flag('rival_stolen_pokemon') and self._check_event_flag('got_starter_pokeball')
        if "capture Hoothoot" in task:
            return self._has_pokemon(HOOTHOOT)
        if "Clear Sprout Tower... receive HM05 Flash" in task:
            # 프록시: 모다피의 탑을 클리어하면 라이벌을 만나고, 다음 관장에게 도전 가능
            return self._check_event_flag('rival_met_sprout_tower')
        if "defeat Falkner, receive Zephyr Badge" in task:
            return self._check_badges(1)
        if "capture Mareep and a Slowpoke" in task:
            return self._has_pokemon(MAREEP) and self._has_pokemon(SLOWPOKE)
        if "Clear Slowpoke Well" in task:
            return self._check_event_flag('rocket_slowpoke_well_defeated')
        if "defeat Bugsy, receive Hive Badge" in task:
            return self._check_badges(2)
        if "receive HM01 Cut" in task:
             # 프록시: 파오리(Farfetch'd)를 주인에게 데려다주면 풀베기를 받음
            return self._check_event_flag('farfetchd_brought_back')
        if "defeat Whitney, receive Plain Badge" in task:
            return self._check_badges(3)
        if "Clear Sudowoodo" in task:
            return self._check_event_flag('battled_sudowoodo')
        if "receive HM03 Surf" in task:
            # 프록시: 인주시티 무용수를 모두 이기면 담청시티 체육관 관장이 등대로 감
            return self._check_event_flag('olivine_gym_leader_lighthouse')
        if "defeat Morty, receive Fog Badge" in task:
            return self._check_badges(4)
        if "get HM04 Strength" in task:
            # hm04_olivine 플래그는 '아직 받지 못함'이므로 False가 되어야 받은 것임
            return not self._check_event_flag('hm04_olivine')
        if "defeat Chuck, receive Storm Badge" in task:
            return self._check_badges(5)
        if "heal Ampharos at the Lighthouse" in task:
            return self._check_event_flag('lighthouse_pokemon_cured')
        if "defeat Jasmine, receive Mineral Badge" in task:
            return self._check_badges(6)
        if "capture the Red Gyarados" in task:
            return self._check_event_flag('battled_red_gyarados')
        if "clear the Team Rocket Hideout" in task:
            return self._check_event_flag('rocket_mahogany_cleared')
        if "defeat Pryce, receive Glacier Badge" in task:
            return self._check_badges(7)
        if "clear the Team Rocket takeover of the Radio Tower" in task:
            return self._check_event_flag('rocket_radio_tower_cleared')
        if "capture Ho-Oh" in task:
            return self._has_pokemon(HO_OH)
        if "defeat Clair, receive Rising Badge" in task:
            return self._check_badges(8)
        if "Complete the Dragon's Den trial" in task:
            # 프록시: 용의 굴에 라이벌이 있다는 것은 시련이 진행 중이거나 끝났다는 의미
            return self._check_event_flag('rival_in_dragons_den')
        if "Defeat Elite Four Will" in task:
            return self._check_event_flag('elite4_will')
        if "Defeat Elite Four Koga" in task:
            return self._check_event_flag('elite4_koga')
        if "Defeat Elite Four Bruno" in task:
            return self._check_event_flag('elite4_bruno')
        if "Defeat Elite Four Karen" in task:
            return self._check_event_flag('elite4_karen')
        if "Defeat Champion Lance" in task:
            return self._check_event_flag('champion_lance')
        if "Become the Johto Champion" in task:
            # 챔피언을 이긴 플래그로 대체
            return self._check_event_flag('champion_lance')

        # 위에서 처리되지 않은 위치 이동 태스크는 현재 위치로 간단히 확인
        # (더 정교한 로직이 필요할 수 있음)
        loc = self.game_state['location']
        if "Violet City" in task and loc['map_bank'] == 3 and loc['map_id'] == 1: return True
        if "Azalea Town" in task and loc['map_bank'] == 4 and loc['map_id'] == 0: return True
        if "Goldenrod City" in task and loc['map_bank'] == 5 and loc['map_id'] == 0: return True
        
        return False # 완료되지 않은 것으로 간주

    def advance_to_next_task(self):
        """다음 목표로 넘어갑니다."""
        if self.current_task_index < len(self.plan['tasks']):
            self.current_task_index += 1
            print(f"***** 다음 목표로 진행합니다: {self.get_current_task_description()} *****")

    def sync_with_initial_state(self, initial_game_state: dict):
        """초기 게임 상태를 확인하고, 이미 완료된 태스크들을 자동으로 건너뜁니다."""
        print("저장된 게임 상태와 목표 리스트를 동기화합니다...")
        while not self.is_current_task_completed(initial_game_state) and self.current_task_index >= len(self.plan['tasks']):
            # while문의 조건 수정: 완료되었을 때 루프를 돌도록
            pass # 이 부분은 로직 재검토 필요
        
        # 수정된 동기화 로직
        while self.is_current_task_completed(initial_game_state):
             if self.current_task_index >= len(self.plan['tasks']): break
             print(f"이미 완료된 목표: '{self.get_current_task_description()}' -> 건너뜁니다.")
             self.advance_to_next_task()
        
        print(f"동기화 완료! 현재 목표: '{self.get_current_task_description()}'")