from pyboy import PyBoy
from pyboy.utils import WindowEvent
import numpy as np

class GameManager:
    """
    PyBoy 에뮬레이터를 관리하고 게임 입력을 처리합니다.
    """
    def __init__(self, rom_path: str, state_path: str = None, speed: int = 0, headless: bool = True):
        self.rom_path = rom_path
        self.state_path = state_path
        
        window = "null" if headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window, sound=False, gameboy_tpye="CGB")
        self.pyboy.set_emulation_speed(speed)
        
        # 가능한 액션과 버튼 릴리즈 매핑
        self.action_map = {
            0: None,
            1: WindowEvent.PRESS_BUTTON_A,
            2: WindowEvent.PRESS_BUTTON_B,
            3: WindowEvent.PRESS_BUTTON_START,
            4: WindowEvent.PRESS_ARROW_UP,
            5: WindowEvent.PRESS_ARROW_DOWN,
            6: WindowEvent.PRESS_ARROW_LEFT,
            7: WindowEvent.PRESS_ARROW_RIGHT,
        }
        self.release_map = {
            0: None,
            1: WindowEvent.RELEASE_BUTTON_A,
            2: WindowEvent.RELEASE_BUTTON_B,
            3: WindowEvent.RELEASE_BUTTON_START,
            4: WindowEvent.RELEASE_ARROW_UP,
            5: WindowEvent.RELEASE_ARROW_DOWN,
            6: WindowEvent.RELEASE_ARROW_LEFT,
            7: WindowEvent.RELEASE_ARROW_RIGHT,
        }

    def reset(self):
        """환경을 초기 상태로 리셋합니다."""
        if self.state_path:
            with open(self.state_path, "rb") as f:
                self.pyboy.load_state(f)
            # 상태 로드 후 안정화를 위해 몇 프레임 진행
            for _ in range(10):
                self.pyboy.tick()
        else:
            # 초기 상태 파일이 없다면 인트로 스킵 (시간이 걸릴 수 있음)
            for _ in range(4000):
                self.pyboy.tick()

    def step(self, action: int, frame_skip: int = 4):
        """
        주어진 액션을 실행하고 지정된 프레임만큼 게임을 진행합니다.
        """
        if self.action_map[action] is not None:
            self.pyboy.send_input(self.action_map[action])
        
        # 버튼을 누른 상태로 몇 프레임 진행
        for _ in range(frame_skip):
            self.pyboy.tick()
        
        if self.release_map[action] is not None:
            self.pyboy.send_input(self.release_map[action])

        # ✨ [핵심 수정] 버튼을 뗀 후, 화면이 렌더링될 시간을 주기 위해 tick을 추가합니다.
        # 이 과정을 통해 다음 화면 캡처 시 온전한 게임 화면을 얻을 수 있습니다.
        for _ in range(frame_skip):
            self.pyboy.tick()

    def get_screen_image(self) -> np.ndarray:
        """
        현재 게임 화면을 흑백(Grayscale) numpy 배열로 반환합니다.
        <<< 최종 수정: pyboy.screen.ndarray -> pyboy.screen_image() >>>
        """
        # PIL Image 객체를 가져와서 numpy 배열로 변환
        screen_pil = self.pyboy.screen_image() 
        screen = np.array(screen_pil)

        if screen.shape[2] > 1:
            grayscale_screen = np.mean(screen[:, :, :3], axis=2).astype(np.uint8)
        else:
            grayscale_screen = screen.squeeze(-1)

        return np.expand_dims(grayscale_screen, axis=-1)
        
    def stop(self):
        """에뮬레이터를 종료합니다."""
        self.pyboy.stop()
    
    def save_state(self, path: str):
        """현재 게임 상태를 지정된 경로에 파일로 저장합니다."""
        with open(path, "wb") as f:
            self.pyboy.save_state(f)