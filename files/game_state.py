# game_state.py (공식 플래그 시스템으로 완전 재작성된 버전)
import numpy as np
from pyboy import PyBoy
from dataclasses import dataclass

# =================================
# 1. 플레이어 상태 및 위치
# =================================
PLAYER_ADDRS = {
    'map_bank': 0xDA00,
    'map_id': 0xDA01,
    'x_coord': 0xDA02,  # X coordinates (2 bytes)
    'y_coord': 0xDA03,  # Y coordinates (2 bytes)
    'x_coord_detailed': 0xD20D,  # Detailed X position
    'y_coord_detailed': 0xD20E,  # Detailed Y position
    'player_id': 0xD1A1,  # 2 bytes
    'player_name': 0xD1A3,  # 10 bytes
    'rival_name': 0xD1BC,   # 10 bytes
    'money': 0xD573,  # 3 bytes, BCD format
    'johto_badges': 0xD57C,
    'kanto_badges': 0xD57D,
    'time_played': 0xD1EB,  # 5 bytes (Day, Hour, Min, Sec, Frame)
    'sprite': 0xD1FF,
    'clothes': 0xD203,
    'on_bike': 0xD682,
    'repel_steps': 0xD9EB,
}

# =================================
# 2. 파티 포켓몬 정보
# =================================
PARTY_ADDRS = {
    'count': 0xDA22,
    'pokemon_list': 0xDA23,  # 포켓몬 ID 리스트 시작 (6개 + FF 종료)
    'end_marker': 0xDA29,    # End of list marker (FF)
    'data_start': 0xDA2A,    # 첫 번째 포켓몬 데이터 시작
}

# 각 포켓몬 데이터 구조 (48바이트씩)
POKEMON_DATA_SIZE = 48
POKEMON_OFFSETS = {
    'species_id': 0x00,      # $DA2A = Pokemon 1
    'item_held': 0x01,       # $DA2B = Item Held  
    'moves': 0x02,           # $DA2C-$DA2F = Moves 1-4 (4 bytes)
    'id_number': 0x06,       # $DA30-$DA31 = ID Number (2 bytes)
    'experience': 0x08,      # $DA32-$DA34 = Experience (3 bytes)
    'hp_ev': 0x0B,          # $DA35-$DA36 = HP EV (2 bytes)
    'attack_ev': 0x0D,      # $DA37-$DA38 = Attack EV (2 bytes)
    'defense_ev': 0x0F,     # $DA39-$DA3A = Defense EV (2 bytes)
    'speed_ev': 0x11,       # $DA3B-$DA3C = Speed EV (2 bytes)
    'special_ev': 0x13,     # $DA3D-$DA3E = Special EV (2 bytes)
    'attack_defense_iv': 0x15, # $DA3F = Attack/Defense IV
    'speed_special_iv': 0x16,  # $DA40 = Speed/Special IV
    'pp_moves': 0x17,       # $DA41-$DA44 = PP Moves (4 bytes)
    'happiness': 0x1B,      # $DA45 = Happiness/Time for Hatching
    'pokerus': 0x1C,        # $DA46 = PokeRus
    'caught_data': 0x1D,    # $DA47-$DA48 = Caught Data (2 bytes)
    'level': 0x1F,          # $DA49 = Level
    'status': 0x20,         # $DA4A-$DA4B = Status (2 bytes)
    'hp': 0x22,             # $DA4C-$DA4D = HP (2 bytes, big-endian)
    'max_hp': 0x24,         # $DA4E-$DA4F = Max HP (2 bytes, big-endian)
    'attack': 0x26,         # $DA50-$DA51 = Attack (2 bytes, big-endian)
    'defense': 0x28,        # $DA52-$DA53 = Defense (2 bytes, big-endian)
    'speed': 0x2A,          # $DA54-$DA55 = Speed (2 bytes, big-endian)
    'sp_defense': 0x2C,     # $DA56-$DA57 = Special Defense (2 bytes, big-endian)
    'sp_attack': 0x2E,      # $DA58-$DA59 = Special Attack (2 bytes, big-endian)
}

# 포켓몬 이름 (OT)
POKEMON_NAMES_OT = {
    'pokemon_1': 0xDB4A,  # 11 bytes each
    'pokemon_2': 0xDB55,
    'pokemon_3': 0xDB60,
    'pokemon_4': 0xDB6B,
    'pokemon_5': 0xDB76,
    'pokemon_6': 0xDB81,
}

# 포켓몬 닉네임
POKEMON_NICKNAMES = {
    'pokemon_1': 0xDB8C,  # 11 bytes each
    'pokemon_2': 0xDB97,
    'pokemon_3': 0xDBA2,
    'pokemon_4': 0xDBAD,
    'pokemon_5': 0xDBB8,
    'pokemon_6': 0xDBC3,
}

# =================================
# 3. 포켓덱스
# =================================
POKEDEX_ADDRS = {
    'owned_start': 0xDBE4,  # Own Pokedex 1-8부터 시작
    'owned_end': 0xDC03,    # Own Pokedex 249-256까지
    'seen_start': 0xDC04,   # Seen Pokedex 1-8부터 시작
    'seen_end': 0xDC23,     # Seen Pokedex 249-256까지
}

# =================================
# 4. 전투 정보
# =================================
BATTLE_ADDRS = {
    'battle_type': 0xD116,      # Type of Battle (Wild/Gym/etc)
    'enemy_species': 0xD0ED,    # Wild Pokemon Number
    'enemy_level': 0xD0FC,      # Wild Pokemon Level / Enemy Level
    'enemy_status': 0xD0FD,     # Enemy Status
    'enemy_hp': 0xD0FF,         # 2 bytes, big-endian (current HP)
    'enemy_max_hp': 0xD101,     # 2 bytes, big-endian (total HP)
    'enemy_attack': 0xD103,     # 2 bytes, big-endian
    'enemy_defense': 0xD105,    # 2 bytes, big-endian
    'enemy_speed': 0xD107,      # 2 bytes, big-endian
    'enemy_sp_attack': 0xD109,  # 2 bytes, big-endian
    'enemy_sp_defense': 0xD10B, # 2 bytes, big-endian
    'enemy_item': 0xD0F0,       # Enemy Item
    'enemy_moves': 0xD0F1,      # Enemy Move 1-4 (4 bytes)
    'enemy_dvs_1': 0xD0F5,      # Enemy DVs: Attack & Defense
    'enemy_dvs_2': 0xD0F6,      # Enemy DVs: Speed & Special
    'enemy_gender': 0xD119,     # Enemy Sex
    'enemy_types': 0xD127,      # Enemy Type 1-2 (2 bytes)
    'enemy_damage': 0xD141,     # Enemy Damage (2 bytes, big-endian)
}

# Your Pokemon in battle
YOUR_BATTLE_ADDRS = {
    'item_held': 0xCB0D,
    'moves': 0xCB0E,        # Move 1-4 (4 bytes)
    'pp_moves': 0xCB14,     # PP Move 1-4 (4 bytes)
    'status': 0xCB1A,
    'hp': 0xCB1C,          # 2 bytes, big-endian
    'types': 0xCB2A,        # Type 1-2 (2 bytes)
    'substitute': 0xCB49,
    'money_earned': 0xCB65, # 2 bytes, big-endian
    'experience_given': 0xCF7E, # 2 bytes, big-endian
    'current_attack': 0xCBC1,
}

# =================================
# 5. 이벤트 플래그 시스템 (공식 문서 기반)
# =================================

# 플래그 베이스 주소와 매핑 테이블 (공식 문서에서)
FLAGS_BASE_ADDR = 0xD7B7

# 공식 주소-플래그 매핑 테이블 (DataCrystal 문서 기준)
FLAG_ADDR_MAP = {
    0xD7B7: 0x0000,  0xD7B8: 0x0800,  0xD7B9: 0x1000,  0xD7BA: 0x1800,
    0xD7BB: 0x2000,  0xD7BC: 0x2800,  0xD7BD: 0x3000,  0xD7BE: 0x3800,
    0xD7BF: 0x4000,  0xD7C0: 0x4800,  0xD7C1: 0x5000,  0xD7C2: 0x5800,
    0xD7C3: 0x6000,  0xD7C4: 0x6800,  0xD7C5: 0x7000,  0xD7C6: 0x7800,
    0xD7C7: 0x8000,  0xD7C8: 0x8800,  0xD7C9: 0x9000,  0xD7CA: 0x9800,
    0xD7CB: 0xA000,  0xD7CC: 0xA800,  0xD7CD: 0xB000,  0xD7CE: 0xB800,
    0xD7CF: 0xC000,  0xD7D0: 0xC800,  0xD7D1: 0xD000,  0xD7D2: 0xD800,
    0xD7D3: 0xE000,  0xD7D4: 0xE800,  0xD7D5: 0xF000,  0xD7D6: 0xF800,
    0xD7D7: 0x0001,  0xD7D8: 0x0801,  0xD7D9: 0x1001,  0xD7DA: 0x1801,
    0xD7DB: 0x2001,  0xD7DC: 0x2801,  0xD7DD: 0x3001,  0xD7DE: 0x3801,
    0xD7DF: 0x4001,  0xD7E0: 0x4801,  0xD7E1: 0x5001,  0xD7E2: 0x5801,
    0xD7E3: 0x6001,  0xD7E4: 0x6801,  0xD7E5: 0x7001,  0xD7E6: 0x7801,
    0xD7E7: 0x8001,  0xD7E8: 0x8801,  0xD7E9: 0x9001,  0xD7EA: 0x9801,
    0xD7EB: 0xA001,  0xD7EC: 0xA801,  0xD7ED: 0xB001,  0xD7EE: 0xB801,
    0xD7EF: 0xC001,  0xD7F0: 0xC801,  0xD7F1: 0xD001,  0xD7F2: 0xD801,
    0xD7F3: 0xE001,  0xD7F4: 0xE801,  0xD7F5: 0xF001,  0xD7F6: 0xF801,
    0xD7F7: 0x0002,  0xD7F8: 0x0802,  0xD7F9: 0x1002,  0xD7FA: 0x1802,
    0xD7FB: 0x2002,  0xD7FC: 0x2802,  0xD7FD: 0x3002,  0xD7FE: 0x3802,
    0xD7FF: 0x4002,  0xD800: 0x4802,  0xD801: 0x5002,  0xD802: 0x5802,
    0xD803: 0x6002,  0xD804: 0x6802,  0xD805: 0x7002,  0xD806: 0x7802,
    0xD807: 0x8002,  0xD808: 0x8802,  0xD809: 0x9002,  0xD80A: 0x9802,
    0xD80B: 0xA002,  0xD80C: 0xA802,  0xD80D: 0xB002,  0xD80E: 0xB802,
    0xD80F: 0xC002,  0xD810: 0xC802,  0xD811: 0xD002,  0xD812: 0xD802,
    0xD813: 0xE002,  0xD814: 0xE802,  0xD815: 0xF002,  0xD816: 0xF802,
    0xD817: 0x0003,  0xD818: 0x0803,  0xD819: 0x1003,  0xD81A: 0x1803,
    0xD81B: 0x2003,  0xD81C: 0x2803,  0xD81D: 0x3003,  0xD81E: 0x3803,
    0xD81F: 0x4003,  0xD820: 0x4803,  0xD821: 0x5003,  0xD822: 0x5803,
    0xD823: 0x6003,  0xD824: 0x6803,  0xD825: 0x7003,  0xD826: 0x7803,
    0xD827: 0x8003,  0xD828: 0x8803,  0xD829: 0x9003,  0xD82A: 0x9803,
    0xD82B: 0xA003,  0xD82C: 0xA803,  0xD82D: 0xB003,  0xD82E: 0xB803,
    0xD82F: 0xC003,  0xD830: 0xC803,  0xD831: 0xD003,  0xD832: 0xD803,
    0xD833: 0xE003,  0xD834: 0xE803,  0xD835: 0xF003,  0xD836: 0xF803,
    0xD837: 0x0004,  0xD838: 0x0804,  0xD839: 0x1004,  0xD83A: 0x1804,
    0xD83B: 0x2004,  0xD83C: 0x2804,  0xD83D: 0x3004,  0xD83E: 0x3804,
    0xD83F: 0x4004,  0xD840: 0x4804,  0xD841: 0x5004,  0xD842: 0x5804,
    0xD843: 0x6004,  0xD844: 0x6804,  0xD845: 0x7004,  0xD846: 0x7804,
    0xD847: 0x8004,  0xD848: 0x8804,  0xD849: 0x9004,  0xD84A: 0x9804,
    0xD84B: 0xA004,  0xD84C: 0xA804,  0xD84D: 0xB004,  0xD84E: 0xB804,
    0xD84F: 0xC004,  0xD850: 0xC804,  0xD851: 0xD004,  0xD852: 0xD804,
    0xD853: 0xE004,  0xD854: 0xE804,  0xD855: 0xF004,  0xD856: 0xF804,
    0xD857: 0x0005,  0xD858: 0x0805,  0xD859: 0x1005,  0xD85A: 0x1805,
    0xD85B: 0x2005,  0xD85C: 0x2805,  0xD85D: 0x3005,  0xD85E: 0x3805,
    0xD85F: 0x4005,  0xD860: 0x4805,  0xD861: 0x5005,  0xD862: 0x5805,
    0xD863: 0x6005,  0xD864: 0x6805,  0xD865: 0x7005,  0xD866: 0x7805,
    0xD867: 0x8005,  0xD868: 0x8805,  0xD869: 0x9005,  0xD86A: 0x9805,
    0xD86B: 0xA005,  0xD86C: 0xA805,  0xD86D: 0xB005,  0xD86E: 0xB805,
    0xD86F: 0xC005,  0xD870: 0xC805,  0xD871: 0xD005,  0xD872: 0xD805,
    0xD873: 0xE005,  0xD874: 0xE805,  0xD875: 0xF005,  0xD876: 0xF805,
    0xD877: 0x0006,  0xD878: 0x0806,  0xD879: 0x1006,  0xD87A: 0x1806,
    0xD87B: 0x2006,  0xD87C: 0x2806,  0xD87D: 0x3006,  0xD87E: 0x3806,
    0xD87F: 0x4006,  0xD880: 0x4806,  0xD881: 0x5006,  0xD882: 0x5806,
    0xD883: 0x6006,  0xD884: 0x6806,  0xD885: 0x7006,  0xD886: 0x7806,
    0xD887: 0x8006,  0xD888: 0x8806,  0xD889: 0x9006,  0xD88A: 0x9806,
    0xD88B: 0xA006,  0xD88C: 0xA806,  0xD88D: 0xB006,  0xD88E: 0xB806,
    0xD88F: 0xC006,  0xD890: 0xC806,  0xD891: 0xD006,  0xD892: 0xD806,
    0xD893: 0xE006,  0xD894: 0xE806,  0xD895: 0xF006,  0xD896: 0xF806,
    0xD897: 0x0007,  0xD898: 0x0807,  0xD899: 0x1007,  0xD89A: 0x1807,
    0xD89B: 0x2007,  0xD89C: 0x2807,  0xD89D: 0x3007,  0xD89E: 0x3807,
    0xD89F: 0x4007,  0xD8A0: 0x4807,  0xD8A1: 0x5007,  0xD8A2: 0x5807,
    0xD8A3: 0x6007,  0xD8A4: 0x6807,  0xD8A5: 0x7007,  0xD8A6: 0x7807,
    0xD8A7: 0x8007,  0xD8A8: 0x8807,  0xD8A9: 0x9007,  0xD8AA: 0x9807,
    0xD8AB: 0xA007,  0xD8AC: 0xA807,  0xD8AD: 0xB007,  0xD8AE: 0xB807,
    0xD8AF: 0xC007,  0xD8B0: 0xC807,  0xD8B1: 0xD007,  0xD8B2: 0xD807,
    0xD8B3: 0xE007,  0xD8B4: 0xE807,  0xD8B5: 0xF007,  0xD8B6: 0xF807,
}

def flag_to_address_bit(flag_id: int) -> tuple[int, int]:
    """
    DataCrystal의 플래그 ID를 실제 메모리 주소와 비트 번호로 변환합니다.
    (수정된 올바른 계산 로직)
    """
    # 0x1B00, 0xBD06 같은 ID는 레이블이며, 실제 순차 ID는 앞 두 자리(0x1B, 0xBD) 입니다.
    # 이 값을 얻기 위해 256(0x100)으로 나눕니다.
    sequential_id = flag_id // 0x100

    # 이 순차 ID를 사용하여 바이트 오프셋과 비트 위치를 계산합니다.
    byte_offset = sequential_id // 8
    bit_num = sequential_id % 8
    
    address = FLAGS_BASE_ADDR + byte_offset
    return address, bit_num

# 공식 문서 기반 이벤트 플래그 정의
EVENT_FLAGS = {
    # ----- 스타터 포켓몬 -----
    'starter_received': 0x1A00,         # Start Pokémon got
    'starter_cyndaquil': 0x1B00,        # Start Pokémon is Cyndaquil
    'starter_totodile': 0x1C00,         # Start Pokémon is Totodile
    'starter_chikorita': 0x1D00,        # Start Pokémon is Chikorita
    'elm_discovery': 0x1E00,            # Elm asks about Mr Pokemon's discovery
    
    # ----- 중요 아이템 -----
    'pokeballs_cherrygrove': 0x1F00,    # pokeballs being sold in cherrygrove
    'got_pokedex': 0xBD06,              # Player has Pokédex
    'rival_stolen_pokemon': 0xBE06,     # Rival has stolen Pokémon
    'old_rod_route32': 0x1700,          # Old Rod in Route 32 PC
    'good_rod_olivine': 0x1800,         # Good Rod in Olivine
    'bike_from_shop': 0x5B00,           # Bike in bikeshop
    'itemfinder_ecruteak': 0x5A00,      # Haven't gotten Itemfinder in Ecruteak yet
    'charcoal_azalea': 0x1000,          # Didn't get charcoal in Azalea
    'hm04_olivine': 0x1300,             # Haven't gotten HM04 in Olivine yet
    
    # ----- 체육관 관련 -----
    # 체육관은 배지 바이트에서 직접 확인 (0xD57C, 0xD57D)
    
    # ----- Elite 4 관련 -----
    'elite4_will': 0x0A03,              # TOP FOUR Will
    'elite4_koga': 0x0C03,              # TOP FOUR Koga  
    'elite4_bruno': 0x0E03,             # TOP FOUR Bruno
    'elite4_karen': 0x1003,             # TOP FOUR Karen
    'champion_lance': 0x1203,           # TOP FOUR CHAMP Lance
    
    # ----- 로켓단 이벤트 -----
    'rocket_slowpoke_well_defeated': 0xD706,  # Player defeated Final TR in Slowpoke Well
    'rocket_radio_tower_attacked': 0xCE06,    # TR has attacked Radio Tower  
    'rocket_radio_tower_cleared': 0xCF06,     # TR is not in Radio Tower
    'rocket_mahogany_cleared': 0xDC06,        # TR is in Mahogany
    'rocket_goldenrod_cleared': 0xE406,       # Got Team Rocket out of Goldenrod
    'rocket_left_goldenrod': 0xCC06,          # TR left Goldenrod
    'rocket_mahogany_controls': 0xDA06,       # Team Rocket Controls Mahogany Store and Secret Base
    
    # ----- 전설 포켓몬 -----
    'battled_red_gyarados': 0xD406,           # Battled Red Gyarados
    'battled_sudowoodo': 0xF806,              # Player battled Sudowoodo
    
    # ----- 라이벌 만남 -----
    'rival_met_cherrygrove': 0xC206,          # Met Rival in Cherrygrove
    'rival_met_goldenrod_underground': 0xC106, # Met Rival in Goldenrod Underground
    'rival_met_sprout_tower': 0xC406,         # Met Rival in Sprout Tower
    'rival_met_burned_tower': 0xC506,         # Met Rival in Burned Tower
    'rival_in_dragons_den': 0xC606,           # Rival is in Dragons Den
    
    # ----- 기타 중요 이벤트 -----
    'got_starter_pokeball': 0xC806,           # Player has Pokémon
    'first_pokemon_center': 0xC706,           # Player comes down 1st time
    'guide_gent_map': 0xFF06,                 # Guide Gent has given map
    'first_mr_pokemon': 0xC906,               # 1st time in Mr. Pokémon House
    'teacher_in_school': 0xCB06,              # Teacher in school
    'lighthouse_pokemon_ill': 0xD206,         # Lighthouse Pokémon is ill
    'lighthouse_pokemon_cured': 0xD306,       # Lighthouse Pokémon cured
    'olivine_gym_leader_lighthouse': 0xC306,  # Olivine Gym Leader in Lighthouse
    'lance_mahogany_store': 0xD506,           # Lance is in Mahogany Store
    
    # ----- 데이케어 관련 -----
    'daycare_man_outside': 0xE506,            # Day Care Man is Outside
    'daycare_pokemon_1': 0xE606,              # Day Care Pokémon (1)
    'daycare_pokemon_2': 0xE706,              # Day Care Pokémon (2)
    'daycare_pokemon_3': 0xE806,              # Day Care Pokémon (3)
    
    # ----- 파닥지 이벤트 -----
    'farfetchd_pos_1': 0xE906,                # Farfetch'd Position 1
    'farfetchd_pos_2': 0xEA06,                # Farfetch'd Position 2
    'farfetchd_pos_3': 0xEB06,                # Farfetch'd Position 3
    'farfetchd_pos_4': 0xEC06,                # Farfetch'd Position 4
    'farfetchd_pos_5': 0xED06,                # Farfetch'd Position 5
    'farfetchd_pos_6a': 0xEE06,               # Farfetch'd Position 6a
    'farfetchd_pos_6b': 0xEF06,               # Farfetch'd Position 6b
    'farfetchd_pos_7': 0xF006,                # Farfetch'd Position 7
    'farfetchd_pos_8': 0xF106,                # Farfetch'd Position 8
    'farfetchd_pos_end': 0xF206,              # Farfetch'd Position (End)
    'farfetchd_still_forest': 0xF306,         # Farfetch'd still in forest
    'farfetchd_brought_back': 0xF406,         # Farfetch'd brought back
    
    # ----- 커트 관련 -----
    'kurt_in_house': 0xFD06,                  # Kurt is in his house
    'tr_azalea': 0xFA06,                      # TR is in Azalea
    'tr_not_azalea': 0xFB06,                  # TR isnt in Azalea
    'didnt_explore_azalea_well': 0xFC06,      # Player didn't explore Azalea Well
    
    # ----- SS 아쿠아 -----
    'ss_aqua_first_time': 0x0007,             # On SS Aqua for first time
    
    # ----- 일요일 형제 시스템 -----
    'arthur': 0x6600,                         # Arthur
    'arthur_flag2': 0x6700,                   # Arthur Flag 2
    'frieda': 0x6200,                         # Frieda
    'frieda_flag2': 0x6300,                   # Frieda Flag 2
    'tuscany': 0x6400,                        # Tuscany
    'tuscany_flag2': 0x6500,                  # Tuscany Flag 2
    'sunny': 0x6800,                          # Sunny
    'sunny_flag2': 0x6900,                    # Sunny Flag 2
    'wesley': 0x6A00,                         # Wesley
    'wesley_flag2': 0x6B00,                   # Wesley Flag 2
    'santos': 0x6C00,                         # Santos
    'santos_flag2': 0x6D00,                   # Santos Flag 2
    'monica': 0x6E00,                         # Monica
    'monica_flag2': 0x6F00,                   # Monica Flag 2
    
    # ----- TM 관련 -----
    'tm_headbutt_ilex': 0x5F00,               # TM Headbutt from Ilex Forest
    'tm_sandstorm_route27': 0x7500,           # Sandstorm TM Route 27
    'tm_sweet_scent_route34': 0x7A00,         # Sweet Scent TM Route 34 Gate
    'tm13_route40': 0x3E00,                   # TM13 on Route 40
    'tm05_route32': 0x4E00,                   # TM05 on Route 32
    'tm10_lake_rage': 0x5800,                 # TM10 from Lake of Rage
    'tm_power_plant': 0xDF00,                 # Got TM in power plant
    
    # ----- HM 관련 -----
    'hm11_radio_tower': 0x2500,               # HM11 in Radio Tower
    
    # ----- 기타 아이템들 -----
    'pink_bow_radio_tower': 0x2100,           # Pink Bow in Radio Tower
    'berry_route30_house': 0x2700,            # Didn't get berry from Route 30 house
    'mystic_water_cherrygrove': 0x4D00,       # Mystic Water from Cherrygrove
    'hp_up_route_guard': 0x5200,              # Got HP UP From route Guard
    'quick_claw_national_park': 0x5700,       # Got Quick Claw in National Park
    'tyrogue_mt_mortar': 0x6100,              # Blackbelt in Mt. Mortar beaten, Tyrogue got
    
    # ----- 발전소 관련 -----
    'transformer_power_plant': 0xC900,        # Transformer in power plant
    'phone_call_power_plant': 0xCA00,         # Phone call in power plant
    'transformer_brought_back': 0xCD00,       # Transformer for power plant brought back
    
    # ----- 마하가니 볼트로브 -----
    'voltorb_1_mahogany': 0xE006,             # Voltorb 1 in Mahogany fainted
    'voltorb_2_mahogany': 0xE106,             # Voltorb 2 in Mahogany fainted
    'voltorb_3_mahogany': 0xE206,             # Voltorb 3 in Mahogany fainted
    
    # ----- 기타 로케이션 이벤트 -----
    'bianca_beaten': 0x5C00,                  # Bianca beaten/not beaten
    'helped_slowpoke_azalea': 0x5E00,         # Helped slowpoke in Azalea, helped in Ilex Forest, charcoal got
    'tr_azalea_beaten': 0x2B00,               # TR in Azalea beaten, not helped in Ilex Forest
    'bill_eevee': 0x3100,                     # Bill hasn't given you Eevee yet
    'doorkey_radio_tower': 0x4A00,            # Doorkey for radio tower
    'door_underground1': 0x4900,              # Door in underground1
    'goldenrod_mart_3rd_floor': 0x4B00,       # 2.Possibility 3rd floor Goldenrod Mart / Counter1
    'switch_mahogany_underground': 0xE202,     # Switch in Mahogany Town underground1
    
    # ----- 박스 더미 -----
    'left_box_pile_underground3': 0x0503,     # Left box pile in underground3 Goldenrod City there/not there
    'right_box_pile_underground3': 0x0603,    # right box pile in underground3 Goldenrod City there/not there
}

# =================================
# 6. 인벤토리
# =================================
INVENTORY_ADDRS = {
    'item_count': 0xD5B7,
    'items_start': 0xD5B8,      # Item 1부터 시작 (Item, Amount 쌍)
    'key_item_count': 0xD5E1,
    'key_items_start': 0xD5E2,  # Key Items
    'ball_count': 0xD5FC,
    'balls_start': 0xD5FD,      # Ball items
}

# TM/HM
TM_HM_ADDRS = {
    'tms_start': 0xD57E,    # TM's (50개)
    'hms_start': 0xD5B0,    # HM's (7개)
}

# =================================
# 7. 게임 모드 및 상태
# =================================
GAME_STATE_ADDRS = {
    'game_mode': 0xD11E,        # 게임 모드 (추정)
    'wild_pokemon_enabled': 0xD20B,  # Wild Pokemon Battles Enabled?
    'options': 0xD199,          # Options
    'brightness': 0xC1CF,       # Brightness  
    'low_hp_warning': 0xC1A6,   # Low HP warning
}


# =================================
# ✨ 1. 새로운 데이터 클래스 정의 (맵 연결 정보 저장용)
# =================================
@dataclass
class MapConnection:
    """맵 연결 정보를 저장하기 위한 데이터 클래스"""
    direction: str          # "NORTH", "SOUTH", "WEST", "EAST"
    dest_bank: int          # 연결된 맵의 뱅크 ID
    dest_map: int           # 연결된 맵의 ID
    
    # 현재 맵에서 이 출구로 가기 위한 대략적인 목표 좌표
    target_x: int           
    target_y: int

# =================================
# ✨ 2. 새로운 RomMapper 클래스 정의 (ROM 데이터 파싱용)
# =================================
class RomMapper:
    """게임 ROM 파일을 읽고 파싱하여 정적 맵 데이터를 추출하는 클래스"""
    
    # DataCrystal 문서에서 확인된 핵심 상수 주소들
    MAP_BANKS_POINTER_TABLE = 0x28000  # ROM Bank 0x0A, Address 0x4000

    def __init__(self, rom_path: str):
        """ROM 파일을 로드하고 초기화합니다."""
        print("ROM Mapper를 초기화하고 ROM 데이터를 로드합니다...")
        try:
            with open(rom_path, 'rb') as f:
                self.rom_data = f.read()
            print("ROM 데이터 로딩 완료.")
        except FileNotFoundError:
            print(f"오류: ROM 파일 '{rom_path}'를 찾을 수 없습니다.")
            self.rom_data = None

    # --- ROM 데이터 읽기 헬퍼 함수 ---
    def _read_byte(self, address: int) -> int:
        return self.rom_data[address]

    def _read_word_le(self, address: int) -> int:
        """리틀 엔디안으로 2바이트를 읽습니다."""
        return self.rom_data[address] + (self.rom_data[address + 1] << 8)

    def _get_bank_start_addr(self, bank_id: int) -> int:
        """맵 뱅크의 시작 주소를 반환합니다."""
        if bank_id == 0: return 0
        pointer_addr = self.MAP_BANKS_POINTER_TABLE + (bank_id - 1) * 2
        return self._read_word_le(pointer_addr)

    def _get_map_header_addr(self, bank_id: int, map_id: int) -> int:
        """특정 맵의 헤더 시작 주소를 계산합니다."""
        bank_start_addr = self._get_bank_start_addr(bank_id)
        
        # 뱅크 시작 주소에 있는 포인터 테이블에서 맵 헤더 주소를 읽음
        pointer_to_header_list = (bank_id - 1) * 0x4000 + (bank_start_addr - 0x4000)
        
        map_header_pointer = pointer_to_header_list + (map_id - 1) * 2
        map_header_local_addr = self._read_word_le(map_header_pointer)
        
        # 최종 ROM 주소로 변환
        return (bank_id - 1) * 0x4000 + (map_header_local_addr - 0x4000)

    # --- 메인 로직: 맵 연결 정보 추출 ---
    def get_map_connections(self, bank_id: int, map_id: int) -> list[MapConnection]:
        """
        주어진 맵의 모든 출구(연결) 정보를 파싱하여 반환합니다.
        DataCrystal의 ROM map 및 Notes 문서를 기반으로 정교하게 구현되었습니다.
        """
        if self.rom_data is None:
            return []
            
        try:
            header_addr = self._get_map_header_addr(bank_id, map_id)
            
            # 1. 메인 헤더에서 2차 헤더 포인터를 읽습니다.
            sec_header_bank = self._read_byte(header_addr + 0)
            sec_header_local_addr = self._read_word_le(header_addr + 4)
            sec_header_addr = (sec_header_bank - 1) * 0x4000 + (sec_header_local_addr - 0x4000)
            
            # 2. 2차 헤더에서 맵 크기와 연결 플래그를 읽습니다.
            map_height = self._read_byte(sec_header_addr + 1)
            map_width = self._read_byte(sec_header_addr + 2)
            connection_flags = self._read_byte(sec_header_addr + 12)
            
            connections = []
            connection_directions = ["NORTH", "SOUTH", "WEST", "EAST"]
            
            # 연결 데이터는 2차 헤더 바로 다음에 위치합니다. (크기 13바이트)
            current_connection_addr = sec_header_addr + 13
            
            # 3. 연결 플래그를 확인하며 각 방향의 연결 정보를 파싱합니다.
            for i, direction in enumerate(connection_directions):
                # 플래그 비트 순서: 0-0-0-0-NORTH-SOUTH-WEST-EAST
                flag_bit = 3 - i
                if (connection_flags >> flag_bit) & 1:
                    # 이 방향의 연결이 존재함. 11바이트 데이터를 읽습니다.
                    dest_bank = self._read_byte(current_connection_addr)
                    dest_map = self._read_byte(current_connection_addr + 1)
                    
                    # Notes 문서에 따르면, offset 8, 9 바이트가 연결 스트립의
                    # x,y 좌표와 관련이 깊어 목표 좌표 추론에 사용합니다.
                    conn_y = self._read_byte(current_connection_addr + 8)
                    conn_x = self._read_byte(current_connection_addr + 9)
                    
                    target_x, target_y = 0, 0
                    if direction == "NORTH":
                        target_y = 0
                        target_x = conn_x
                    elif direction == "SOUTH":
                        target_y = map_height + 3 # 맵 경계 밖
                        target_x = conn_x
                    elif direction == "WEST":
                        target_x = 0
                        target_y = conn_y
                    elif direction == "EAST":
                        target_x = map_width + 3 # 맵 경계 밖
                        target_y = conn_y

                    connections.append(MapConnection(
                        direction=direction,
                        dest_bank=dest_bank,
                        dest_map=dest_map,
                        target_x=target_x,
                        target_y=target_y,
                    ))
                    
                    # 다음 연결 데이터 주소로 이동 (11바이트)
                    current_connection_addr += 11

            return connections
        except Exception as e:
            # print(f"맵 연결 정보 파싱 중 오류 발생 (Bank: {bank_id}, Map: {map_id}): {e}")
            return []


# =================================
# GameState 클래스
# =================================
class GameState:
    def __init__(self, pyboy: PyBoy, rom_path: str = "PokemonGold.gbc"): 
        self.pyboy = pyboy
        # ✨ RomMapper 인스턴스를 생성합니다.
        self.rom_mapper = RomMapper(rom_path)

    # --- 메모리 읽기 ---
    def _read_memory(self, address: int) -> int:
        """메모리에서 1바이트 읽기"""
        return self.pyboy.get_memory_value(address)

    def _read_word_big_endian(self, address: int) -> int:
        """2바이트를 빅엔디안으로 읽기"""
        return (self._read_memory(address) << 8) + self._read_memory(address + 1)
    
    def _read_word_little_endian(self, address: int) -> int:
        """2바이트를 리틀엔디안으로 읽기"""
        return self._read_memory(address) + (self._read_memory(address + 1) << 8)

    def _read_bcd(self, address: int, num_bytes: int) -> int:
        """BCD 형식으로 읽기 (돈 등)"""
        value = 0
        for i in range(num_bytes):
            byte = self._read_memory(address + i)
            value = value * 100 + ((byte >> 4) * 10 + (byte & 0x0F))
        return value

    def _read_string(self, start_address: int, max_len: int = 10) -> str:
        """문자열 읽기 (0x50이 종료 문자)"""
        chars = []
        for i in range(max_len):
            char_code = self._read_memory(start_address + i)
            if char_code == 0x50:  # 종료 문자
                break
            # 간단한 문자 매핑 (실제로는 더 복잡한 매핑 필요)
            if 0x80 <= char_code <= 0x99:  # A-Z
                chars.append(chr(ord('A') + char_code - 0x80))
            elif 0x9A <= char_code <= 0xB3:  # a-z  
                chars.append(chr(ord('a') + char_code - 0x9A))
            elif char_code == 0x7F:  # 공백
                chars.append(' ')
            else:
                chars.append(f'[{char_code:02X}]')
        return "".join(chars)

    # --- 플래그 체크 (공식 방식) ---
    def _check_flag(self, flag_id: int) -> bool:
        """
        공식 플래그 시스템으로 플래그 상태 확인
        
        Args:
            flag_id: 플래그 ID (예: 0x1A00)
        
        Returns:
            bool: 플래그가 설정되어 있으면 True
        """
        try:
            addr, bit = flag_to_address_bit(flag_id)
            
            # 주소 범위 체크
            if addr > 0xFFFF or addr < 0x8000:
                return False
                
            byte_val = self._read_memory(addr)
            return (byte_val & (1 << bit)) != 0
            
        except Exception:
            return False

    def check_multiple_flags(self, flag_ids: list[int]) -> dict[int, bool]:
        """여러 플래그를 한 번에 체크"""
        return {flag_id: self._check_flag(flag_id) for flag_id in flag_ids}

    def get_flag_details(self, flag_id: int) -> dict:
        """플래그의 상세 정보 (디버깅용)"""
        addr, bit = flag_to_address_bit(flag_id)
        
        if addr > 0xFFFF or addr < 0x8000:
            return {
                'flag_id': f"0x{flag_id:04X}",
                'address': f"0x{addr:04X}",
                'bit': bit,
                'valid': False,
                'value': False,
                'byte_value': 0
            }
        
        byte_val = self._read_memory(addr)
        flag_val = (byte_val & (1 << bit)) != 0
        
        return {
            'flag_id': f"0x{flag_id:04X}",
            'address': f"0x{addr:04X}",
            'bit': bit,
            'bit_mask': f"0x{1 << bit:02X}",
            'valid': True,
            'value': flag_val,
            'byte_value': f"0x{byte_val:02X}"
        }

    # --- 플레이어 정보 ---
    def _get_player_info(self) -> dict:
        johto_badges_byte = self._read_memory(PLAYER_ADDRS['johto_badges'])
        kanto_badges_byte = self._read_memory(PLAYER_ADDRS['kanto_badges'])
        
        return {
            'player_id': self._read_word_little_endian(PLAYER_ADDRS['player_id']),
            'player_name': self._read_string(PLAYER_ADDRS['player_name']),
            'rival_name': self._read_string(PLAYER_ADDRS['rival_name']),
            'money': self._read_bcd(PLAYER_ADDRS['money'], 3),
            'johto_badges_byte': johto_badges_byte,
            'johto_badges_count': bin(johto_badges_byte).count('1'),
            'kanto_badges_byte': kanto_badges_byte,
            'kanto_badges_count': bin(kanto_badges_byte).count('1'),
            'sprite': self._read_memory(PLAYER_ADDRS['sprite']),
            'clothes': self._read_memory(PLAYER_ADDRS['clothes']),
            'on_bike': self._read_memory(PLAYER_ADDRS['on_bike']) != 0,
            'repel_steps': self._read_memory(PLAYER_ADDRS['repel_steps']),
        }

    def _get_location_info(self) -> dict:
        return {
            'map_bank': self._read_memory(PLAYER_ADDRS['map_bank']),
            'map_id': self._read_memory(PLAYER_ADDRS['map_id']),
            'x_coord': self._read_memory(PLAYER_ADDRS['x_coord']),
            'y_coord': self._read_memory(PLAYER_ADDRS['y_coord']),
            'x_coord_detailed': self._read_memory(PLAYER_ADDRS['x_coord_detailed']),
            'y_coord_detailed': self._read_memory(PLAYER_ADDRS['y_coord_detailed']),
        }

    # --- 파티 정보 ---
    def _get_party_info(self) -> list[dict]:
        party_count = self._read_memory(PARTY_ADDRS['count'])
        party = []
        
        for i in range(min(party_count, 6)):  # 최대 6마리
            start_addr = PARTY_ADDRS['data_start'] + i * POKEMON_DATA_SIZE
            
            # 기본 정보
            species_id = self._read_memory(start_addr + POKEMON_OFFSETS['species_id'])
            level = self._read_memory(start_addr + POKEMON_OFFSETS['level'])
            
            # HP 정보
            current_hp = self._read_word_big_endian(start_addr + POKEMON_OFFSETS['hp'])
            max_hp = self._read_word_big_endian(start_addr + POKEMON_OFFSETS['max_hp'])
            
            # 스탯 정보
            attack = self._read_word_big_endian(start_addr + POKEMON_OFFSETS['attack'])
            defense = self._read_word_big_endian(start_addr + POKEMON_OFFSETS['defense'])
            speed = self._read_word_big_endian(start_addr + POKEMON_OFFSETS['speed'])
            sp_attack = self._read_word_big_endian(start_addr + POKEMON_OFFSETS['sp_attack'])
            sp_defense = self._read_word_big_endian(start_addr + POKEMON_OFFSETS['sp_defense'])
            
            # 경험치 (3바이트)
            exp_addr = start_addr + POKEMON_OFFSETS['experience']
            experience = (self._read_memory(exp_addr) << 16) + \
                        (self._read_memory(exp_addr + 1) << 8) + \
                        self._read_memory(exp_addr + 2)
            
            # 개체값
            iv_byte1 = self._read_memory(start_addr + POKEMON_OFFSETS['attack_defense_iv'])
            iv_byte2 = self._read_memory(start_addr + POKEMON_OFFSETS['speed_special_iv'])
            
            attack_iv = (iv_byte1 >> 4) & 0xF
            defense_iv = iv_byte1 & 0xF
            speed_iv = (iv_byte2 >> 4) & 0xF
            special_iv = iv_byte2 & 0xF
            
            # 기술들
            moves = []
            for j in range(4):
                move = self._read_memory(start_addr + POKEMON_OFFSETS['moves'] + j)
                moves.append(move)
            
            # 기술 PP
            pp_moves = []
            for j in range(4):
                pp = self._read_memory(start_addr + POKEMON_OFFSETS['pp_moves'] + j)
                pp_moves.append(pp)
            
            pokemon = {
                'species_id': species_id,
                'level': level,
                'current_hp': current_hp,
                'max_hp': max_hp,
                'attack': attack,
                'defense': defense, 
                'speed': speed,
                'sp_attack': sp_attack,
                'sp_defense': sp_defense,
                'experience': experience,
                'ivs': {
                    'attack': attack_iv,
                    'defense': defense_iv,
                    'speed': speed_iv,
                    'special': special_iv
                },
                'moves': moves,
                'pp_moves': pp_moves,
                'happiness': self._read_memory(start_addr + POKEMON_OFFSETS['happiness']),
                'pokerus': self._read_memory(start_addr + POKEMON_OFFSETS['pokerus']),
                'status': self._read_word_big_endian(start_addr + POKEMON_OFFSETS['status']),
                'item_held': self._read_memory(start_addr + POKEMON_OFFSETS['item_held']),
            }
            party.append(pokemon)
        
        return party

    # --- 이벤트 플래그 (공식 시스템 사용) ---
    def _get_event_flags_info(self) -> dict:
        """모든 정의된 이벤트 플래그 상태 확인"""
        flags = {}
        for name, flag_id in EVENT_FLAGS.items():
            flags[name] = self._check_flag(flag_id)
        return flags

    def get_starter_info(self) -> dict:
        """스타터 포켓몬 관련 이벤트 요약 (공식 플래그 사용)"""
        # 공식 플래그로 먼저 확인
        starter_received = self._check_flag(EVENT_FLAGS['starter_received'])
        cyndaquil = self._check_flag(EVENT_FLAGS['starter_cyndaquil'])
        totodile = self._check_flag(EVENT_FLAGS['starter_totodile'])  
        chikorita = self._check_flag(EVENT_FLAGS['starter_chikorita'])
        
        starter = None
        if cyndaquil:
            starter = "Cyndaquil"
        elif totodile:
            starter = "Totodile"
        elif chikorita:
            starter = "Chikorita"
        
        # 파티에 포켓몬이 있는지로도 판단
        party_count = self._read_memory(PARTY_ADDRS['count'])
        has_pokemon = self._check_flag(EVENT_FLAGS['got_starter_pokeball'])
        
        # 백업 방법: 파티에서 첫 포켓몬 확인
        if party_count > 0 and not starter:
            first_species = self._read_memory(PARTY_ADDRS['data_start'])
            if first_species == 155:  # Cyndaquil
                starter = "Cyndaquil"
            elif first_species == 158:  # Totodile
                starter = "Totodile"
            elif first_species == 152:  # Chikorita
                starter = "Chikorita"
            
        return {
            "received": starter_received or has_pokemon or (party_count > 0),
            "which": starter,
            "flag_details": {
                "starter_received": starter_received,
                "cyndaquil_flag": cyndaquil,
                "totodile_flag": totodile,
                "chikorita_flag": chikorita,
                "has_pokemon_flag": has_pokemon,
                "party_count": party_count
            }
        }

    def get_rocket_events_info(self) -> dict:
        """로켓단 관련 이벤트들 요약"""
        return {
            "slowpoke_well_defeated": self._check_flag(EVENT_FLAGS['rocket_slowpoke_well_defeated']),
            "radio_tower_attacked": self._check_flag(EVENT_FLAGS['rocket_radio_tower_attacked']),
            "radio_tower_cleared": self._check_flag(EVENT_FLAGS['rocket_radio_tower_cleared']),
            "mahogany_cleared": self._check_flag(EVENT_FLAGS['rocket_mahogany_cleared']),
            "goldenrod_cleared": self._check_flag(EVENT_FLAGS['rocket_goldenrod_cleared']),
            "left_goldenrod": self._check_flag(EVENT_FLAGS['rocket_left_goldenrod']),
            "mahogany_controls": self._check_flag(EVENT_FLAGS['rocket_mahogany_controls']),
        }

    def get_elite4_info(self) -> dict:
        """Elite 4 진행 상황"""
        return {
            "will_defeated": self._check_flag(EVENT_FLAGS['elite4_will']),
            "koga_defeated": self._check_flag(EVENT_FLAGS['elite4_koga']),
            "bruno_defeated": self._check_flag(EVENT_FLAGS['elite4_bruno']),
            "karen_defeated": self._check_flag(EVENT_FLAGS['elite4_karen']),
            "lance_defeated": self._check_flag(EVENT_FLAGS['champion_lance']),
        }

    def get_legendary_pokemon_info(self) -> dict:
        """전설 포켓몬 관련 정보"""
        return {
            "red_gyarados_battled": self._check_flag(EVENT_FLAGS['battled_red_gyarados']),
            "sudowoodo_battled": self._check_flag(EVENT_FLAGS['battled_sudowoodo']),
        }

    def get_rival_encounters_info(self) -> dict:
        """라이벌과의 만남 정보"""
        return {
            "met_cherrygrove": self._check_flag(EVENT_FLAGS['rival_met_cherrygrove']),
            "met_goldenrod_underground": self._check_flag(EVENT_FLAGS['rival_met_goldenrod_underground']),
            "met_sprout_tower": self._check_flag(EVENT_FLAGS['rival_met_sprout_tower']),
            "met_burned_tower": self._check_flag(EVENT_FLAGS['rival_met_burned_tower']),
            "in_dragons_den": self._check_flag(EVENT_FLAGS['rival_in_dragons_den']),
            "stolen_pokemon": self._check_flag(EVENT_FLAGS['rival_stolen_pokemon']),
        }

    # --- 전투 정보 ---
    def _get_battle_info(self) -> dict:
        battle_type = self._read_memory(BATTLE_ADDRS['battle_type'])
        if battle_type == 0:
            return None
            
        return {
            'battle_type': battle_type,
            'enemy_species': self._read_memory(BATTLE_ADDRS['enemy_species']),
            'enemy_level': self._read_memory(BATTLE_ADDRS['enemy_level']),
            'enemy_hp': self._read_word_big_endian(BATTLE_ADDRS['enemy_hp']),
            'enemy_max_hp': self._read_word_big_endian(BATTLE_ADDRS['enemy_max_hp']),
            'enemy_status': self._read_memory(BATTLE_ADDRS['enemy_status']),
        }

    # --- 인벤토리 정보 ---
    def _get_inventory_info(self) -> dict:
        item_count = self._read_memory(INVENTORY_ADDRS['item_count'])
        items = []
        
        for i in range(min(item_count, 20)):  # 최대 20개 아이템
            item_addr = INVENTORY_ADDRS['items_start'] + i * 2
            item_id = self._read_memory(item_addr)
            item_amount = self._read_memory(item_addr + 1)
            items.append({'id': item_id, 'amount': item_amount})
        
        return {
            'item_count': item_count,
            'items': items,
        }

    # --- 상태 체크 ---
    def is_in_menu(self) -> bool:
        # 게임 모드 체크 (정확한 값은 테스트 필요)
        game_mode = self._read_memory(GAME_STATE_ADDRS.get('game_mode', 0xD11E))
        return game_mode == 2

    def is_in_battle(self) -> bool:
        battle_type = self._read_memory(BATTLE_ADDRS['battle_type'])
        return battle_type != 0

    # --- 포켓덱스 정보 ---
    def get_pokedex_info(self) -> dict:
        """포켓덱스 정보 (seen/owned 개수)"""
        owned_count = 0
        seen_count = 0
        
        # Owned 개수 계산
        for addr in range(POKEDEX_ADDRS['owned_start'], POKEDEX_ADDRS['owned_end'] + 1):
            byte_val = self._read_memory(addr)
            owned_count += bin(byte_val).count('1')
        
        # Seen 개수 계산    
        for addr in range(POKEDEX_ADDRS['seen_start'], POKEDEX_ADDRS['seen_end'] + 1):
            byte_val = self._read_memory(addr)
            seen_count += bin(byte_val).count('1')
            
        return {
            'owned_count': owned_count,
            'seen_count': seen_count,
        }

    # ✨ 현재 맵의 연결 정보를 가져오는 새 메서드 추가
    def _get_current_map_connections(self) -> list[dict]:
        """현재 위치한 맵의 출구 정보를 ROM에서 읽어옵니다."""
        loc = self._get_location_info()
        connections_data = self.rom_mapper.get_map_connections(loc['map_bank'], loc['map_id'])
        # dataclass 객체를 일반 dict로 변환하여 반환
        return [c.__dict__ for c in connections_data]
    


    # --- 최종 상태 dict ---
    def get_state_dict(self) -> dict:
        party_list = self._get_party_info()
        battle_info = self._get_battle_info()
        is_battle = battle_info is not None

        party_level_sum = sum(p.get('level',0) for p in party_list)
        party_hp_sum = sum(p.get('current_hp',0) for p in party_list)
        
        state = {
            'is_in_battle': is_battle,
            'is_in_menu': self.is_in_menu(),
            'location': self._get_location_info(),
            'player_info': self._get_player_info(),
            'party_info': {
                'count': len(party_list),
                'pokemon': party_list,
                'party_level_sum': party_level_sum,
                'party_hp_sum': party_hp_sum,
            },
            # ✨ 최종 상태에 맵 연결 정보를 추가합니다.
            'map_connections': self._get_current_map_connections(),
            'event_statuses': self._get_event_flags_info(),
            'starter_info': self.get_starter_info(),
            'rocket_events': self.get_rocket_events_info(),
            'elite4_progress': self.get_elite4_info(),
            'legendary_pokemon': self.get_legendary_pokemon_info(),
            'rival_encounters': self.get_rival_encounters_info(),
            'pokedex': self.get_pokedex_info(),
            'inventory': self._get_inventory_info(),
        }

        if is_battle:
            state['battle_info'] = battle_info

        return state

    # --- 디버그용 함수들 ---
    def debug_memory_range(self, start_addr: int, end_addr: int) -> dict:
        """특정 메모리 범위의 값들을 확인 (디버깅용)"""
        data = {}
        for addr in range(start_addr, end_addr + 1):
            data[f"0x{addr:04X}"] = self._read_memory(addr)
        return data
    
    def debug_flag_system(self) -> dict:
        """플래그 시스템 전체 디버그 정보"""
        debug_info = {
            'flag_base_address': f"0x{FLAGS_BASE_ADDR:04X}",
            'sample_flags': {},
            'flag_memory_state': {},
        }
        
        # 몇 가지 중요한 플래그들의 상세 정보
        important_flags = [
            'starter_received', 'starter_cyndaquil', 'starter_totodile', 'starter_chikorita',
            'got_pokedex', 'rival_stolen_pokemon', 'battled_red_gyarados'
        ]
        
        for flag_name in important_flags:
            if flag_name in EVENT_FLAGS:
                flag_id = EVENT_FLAGS[flag_name]
                debug_info['sample_flags'][flag_name] = self.get_flag_details(flag_id)
        
        # 주요 플래그 메모리 영역 상태
        for addr in range(0xD7B7, 0xD7C0):  # 처음 몇 바이트만
            debug_info['flag_memory_state'][f"0x{addr:04X}"] = f"0x{self._read_memory(addr):02X}"
        
        return debug_info

    def debug_specific_flags(self, flag_names: list[str]) -> dict:
        """특정 플래그들의 상세 정보"""
        results = {}
        for flag_name in flag_names:
            if flag_name in EVENT_FLAGS:
                flag_id = EVENT_FLAGS[flag_name]
                results[flag_name] = self.get_flag_details(flag_id)
            else:
                results[flag_name] = {'error': 'Flag not found'}
        return results

    def search_flags_by_pattern(self, pattern: str) -> dict:
        """패턴으로 플래그 검색 (예: 'starter', 'rocket' 등)"""
        matching_flags = {}
        for flag_name, flag_id in EVENT_FLAGS.items():
            if pattern.lower() in flag_name.lower():
                matching_flags[flag_name] = {
                    'flag_id': f"0x{flag_id:04X}",
                    'value': self._check_flag(flag_id),
                    'details': self.get_flag_details(flag_id)
                }