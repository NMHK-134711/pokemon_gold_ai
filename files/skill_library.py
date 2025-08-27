# skill_library.py

from game_state import GameState, EVENT_FLAGS

# ==============================================================================
# 1. 기본 스킬 클래스
# ==============================================================================
class Skill:
    """모든 스킬의 기본이 되는 추상 클래스"""
    def __init__(self, description: str):
        self.description = description

    def is_achieved(self, prev_state: dict, current_state: dict) -> bool:
        """이 스킬의 목표가 달성되었는지 확인합니다."""
        raise NotImplementedError

    def get_reward(self, prev_state: dict, current_state: dict) -> float:
        """스킬 달성 시 즉각적인 큰 보상을 제공합니다."""
        if self.is_achieved(prev_state, current_state) and not self.is_achieved({}, prev_state):
            return 500.0  # 달성 보상
        return 0.0

# ==============================================================================
# 2. 구체적인 스킬 클래스 정의
# ==============================================================================

class GoToMapSkill(Skill):
    """특정 맵으로 이동하는 스킬"""
    def __init__(self, map_bank: int, map_id: int, map_name: str):
        super().__init__(f"{map_name}(으)로 이동하기")
        self.map_bank = map_bank
        self.map_id = map_id

    def is_achieved(self, prev_state, current_state) -> bool:
        loc = current_state['location']
        return loc['map_bank'] == self.map_bank and loc['map_id'] == self.map_id

class DefeatGymLeaderSkill(Skill):
    """체육관 관장을 이기고 배지를 획득하는 스킬"""
    def __init__(self, badge_name: str, target_badge_count: int):
        super().__init__(f"{badge_name} 배지 획득하기 ({target_badge_count}번째)")
        self.target_badge_count = target_badge_count

    def is_achieved(self, prev_state, current_state) -> bool:
        return current_state['player_info']['johto_badges_count'] >= self.target_badge_count

class CompleteEventFlagSkill(Skill):
    """특정 이벤트 플래그를 달성하는 스킬 (가장 일반적인 스토리 진행)"""
    def __init__(self, event_name: str, description: str):
        super().__init__(description)
        self.event_name = event_name

    def is_achieved(self, prev_state, current_state) -> bool:
        return current_state['event_statuses'].get(self.event_name, False)

class ObtainItemSkill(Skill):
    """특정 아이템을 획득하는 스킬"""
    def __init__(self, item_id: int, item_name: str, item_type: str = 'items'):
        super().__init__(f"{item_name} 획득하기")
        self.item_id = item_id
        self.item_type = item_type # 'items', 'key_items', 'balls'

    def is_achieved(self, prev_state, current_state) -> bool:
        # 이 기능은 GameState 클래스에 인벤토리 읽기 기능이 추가되어야 완벽히 동작합니다.
        # 여기서는 GameState가 인벤토리 정보를 제공한다고 가정합니다.
        inventory = current_state.get('inventory', {})
        item_list = inventory.get(self.item_type, [])
        return any(item['id'] == self.item_id for item in item_list)

class CapturePokemonSkill(Skill):
    """특정 포켓몬을 포획하는 스킬"""
    def __init__(self, species_id: int, species_name: str):
        super().__init__(f"{species_name} 포획하기")
        self.species_id = species_id

    def is_achieved(self, prev_state, current_state) -> bool:
        party = current_state['party_info']['pokemon']
        return any(p['species_id'] == self.species_id for p in party)

class LevelUpSkill(Skill):
    """파티 포켓몬의 레벨을 올리는 스킬"""
    def __init__(self, target_level: int):
        super().__init__(f"파티 포켓몬 중 하나의 레벨을 {target_level} 이상으로 올리기")
        self.target_level = target_level

    def is_achieved(self, prev_state, current_state) -> bool:
        party = current_state['party_info']['pokemon']
        if not party: return False
        return any(p['level'] >= self.target_level for p in party)

# ==============================================================================
# 3. ✨ 사용 가능한 모든 스킬 목록 (게임 계획 기반) ✨
# ==============================================================================
AVAILABLE_SKILLS = [
    # --- 1. 연두마을 ~ 도라지시티 ---
    GoToMapSkill(map_bank=24, map_id=5, map_name="29번 도로"),
    GoToMapSkill(map_bank=24, map_id=7, map_name="무궁시티"),
    GoToMapSkill(map_bank=24, map_id=9, map_name="30번 도로"),
    CompleteEventFlagSkill('got_pokedex', "포켓몬 도감과 알 획득하기"),
    GoToMapSkill(map_bank=24, map_id=4, map_name="연두마을 (복귀)"),
    CapturePokemonSkill(species_id=163, species_name="부우부"), # Hoothoot
    GoToMapSkill(map_bank=3, map_id=1, map_name="도라지시티"),
    # 프록시: 모다피의 탑을 클리어하면 라이벌을 만남
    CompleteEventFlagSkill('rival_met_sprout_tower', "모다피의 탑 클리어하기"),
    DefeatGymLeaderSkill(badge_name="윙", target_badge_count=1),
    
    # --- 2. 도라지시티 ~ 고동마을 ---
    GoToMapSkill(map_bank=24, map_id=12, map_name="32번 도로"),
    CapturePokemonSkill(species_id=179, species_name="메리프"), # Mareep
    CapturePokemonSkill(species_id=79, species_name="야돈"), # Slowpoke
    GoToMapSkill(map_bank=4, map_id=0, map_name="고동마을"),
    CompleteEventFlagSkill('rocket_slowpoke_well_defeated', "고동우물 로켓단 소탕하기"),
    DefeatGymLeaderSkill(badge_name="하이브", target_badge_count=2),

    # --- 3. 고동마을 ~ 금빛시티 ---
    # 프록시: 파오리를 데려다주면 풀베기를 받음
    CompleteEventFlagSkill('farfetchd_brought_back', "비전머신01 풀베기 획득하기"),
    GoToMapSkill(map_bank=5, map_id=0, map_name="금빛시티"),
    DefeatGymLeaderSkill(badge_name="플레인", target_badge_count=3),

    # --- 4. 금빛시티 ~ 인주시티 ---
    GoToMapSkill(map_bank=6, map_id=0, map_name="인주시티"),
    # 프록시: 무용수를 이기면 담청시티 관장이 등대로 감
    CompleteEventFlagSkill('olivine_gym_leader_lighthouse', "인주시티 무용수 전원 격파하기"),
    DefeatGymLeaderSkill(badge_name="팬텀", target_badge_count=4),

    # --- 5. 인주시티 ~ 진청시티 ---
    GoToMapSkill(map_bank=17, map_id=0, map_name="담청시티"),
    GoToMapSkill(map_bank=18, map_id=0, map_name="진청시티"),
    DefeatGymLeaderSkill(badge_name="쇼크", target_badge_count=5),

    # --- 6. 진청시티 ~ 담청시티 체육관 ---
    CompleteEventFlagSkill('lighthouse_pokemon_cured', "담청 등대 포켓몬 치료하기"),
    DefeatGymLeaderSkill(badge_name="스틸", target_badge_count=6),

    # --- 7. 담청시티 ~ 황토마을 ---
    GoToMapSkill(map_bank=19, map_id=0, map_name="황토마을"),
    CompleteEventFlagSkill('battled_red_gyarados', "붉은 갸라도스와 조우/격파하기"),
    CompleteEventFlagSkill('rocket_mahogany_cleared', "황토마을 로켓단 아지트 소탕하기"),
    DefeatGymLeaderSkill(badge_name="아이스", target_badge_count=7),

    # --- 8. 황토마을 ~ 검은먹시티 ---
    CompleteEventFlagSkill('rocket_radio_tower_cleared', "금빛시티 라디오타워 로켓단 소탕하기"),
    GoToMapSkill(map_bank=20, map_id=0, map_name="검은먹시티"),
    # 프록시: 용의 굴에 라이벌이 있음
    CompleteEventFlagSkill('rival_in_dragons_den', "용의 굴 시련 통과하기"),
    DefeatGymLeaderSkill(badge_name="라이징", target_badge_count=8),

    # --- 9. 챔피언 로드 및 사천왕 ---
    GoToMapSkill(map_bank=26, map_id=4, map_name="챔피언 로드"),
    GoToMapSkill(map_bank=10, map_id=1, map_name="석영고원"),
    CompleteEventFlagSkill('elite4_will', "사천왕 일목 격파하기"),
    CompleteEventFlagSkill('elite4_koga', "사천왕 독수 격파하기"),
    CompleteEventFlagSkill('elite4_bruno', "사천왕 시바 격파하기"),
    CompleteEventFlagSkill('elite4_karen', "사천왕 카렌 격파하기"),
    CompleteEventFlagSkill('champion_lance', "챔피언 목호 격파하기"),
    # --- 레벨업 스킬 (중간 목표) ---
    LevelUpSkill(target_level=10),
    LevelUpSkill(target_level=15),
    LevelUpSkill(target_level=20),
    LevelUpSkill(target_level=25),
    LevelUpSkill(target_level=30),
    LevelUpSkill(target_level=35),
    LevelUpSkill(target_level=40),
    LevelUpSkill(target_level=45),
    LevelUpSkill(target_level=50),
]

class HealPartySkill(Skill):
    def __init__(self, heal_location: dict):
        super().__init__(f"{heal_location['name']}에서 파티 회복하기")
        self.heal_location = heal_location

    def is_achieved(self, prev_state: dict, current_state: dict) -> bool:
        # <<< 수정된 방식: 모든 파티원의 HP를 개별적으로 확인 >>>
        if not current_state.get('party_info', {}).get('pokemon'):
            return True  # 파티가 없으면 회복할 필요가 없음

        # 파티의 '모든' 포켓몬의 HP가 95% 이상인지 확인
        return all(
            p.get('max_hp', 0) == 0 or (p.get('hp', 0) / p.get('max_hp', 1)) > 0.95
            for p in current_state['party_info']['pokemon']
        )

    def get_reward(self, prev_state: dict, current_state: dict) -> float:
        # 목표 지점에 가까워질수록 보상
        loc = current_state['location']
        target_loc = self.heal_location
        
        # 맵이 다르면 큰 음수 보상 (잘못된 길)
        if loc['map_bank'] != target_loc['map_bank'] or loc['map_id'] != target_loc['map_id']:
            return -0.1
        
        # 맵이 같으면, 목표 지점과의 거리 차이로 보상 계산
        dist = abs(loc['x_coord'] - target_loc['x']) + abs(loc['y_coord'] - target_loc['y'])
        
        prev_loc = prev_state['location']
        prev_dist = abs(prev_loc['x_coord'] - target_loc['x']) + abs(prev_loc['y_coord'] - target_loc['y'])
        
        # 거리가 줄어들면 양수 보상, 늘어나면 음수 보상
        return (prev_dist - dist) * 0.1