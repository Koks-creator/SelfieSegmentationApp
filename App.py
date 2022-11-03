from threading import Thread
from functools import partial
import os
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.config import Config
import mediapipe as mp
import cv2
import numpy as np


WIDTH = 1280
HEIGHT = 720
BACKGROUNDS_DIR = "Backgrounds"
ALLOWED_EXTS = [".png", ".jpg", ".jpeg"]
BACKGROUNDS = [rf"{BACKGROUNDS_DIR}\{file}" for file in os.listdir(BACKGROUNDS_DIR) if os.path.splitext(file)[1] in ALLOWED_EXTS]

Config.set('graphics', 'width', str(WIDTH))
Config.set('graphics', 'height', str(HEIGHT))

if not os.path.exists(BACKGROUNDS_DIR):
    os.mkdir("Backgrounds")


class MainScreen(Screen):
    container = ObjectProperty(None)
    background_index = 0
    blur = False

    def __init__(self, **kwargs):
        super(Screen, self).__init__(**kwargs)
        Clock.schedule_once(self.setup_scrollview, 1)

    def setup_scrollview(self, dt) -> None:
        self.container.bind(minimum_height=self.container.setter('height'))
        self.create_background_buttons()

    def create_background_buttons(self) -> None:
        for index, file_path in enumerate(BACKGROUNDS):
            btn = Button(
                color=(1, 0, .65, 1),
                background_normal=file_path,
                size_hint_x=None,
                size_hint_y=None,
                width=150,
                border=(0, 1, 1, 0),
                text=str(index),
                font_size='1px',
            )
            btn.bind(on_press=self.get_background_index)
            self.container.add_widget(btn)

    @staticmethod
    def get_background_index(btn: Button) -> None:
        MainScreen.background_index = int(btn.text)

    @staticmethod
    def update_blur() -> None:
        if not MainScreen.blur:
            MainScreen.blur = True
        else:
            MainScreen.blur = False

    @staticmethod
    def change_background_index(mode: int) -> None:
        """
        :param mode: 0 do przodu, 1 do tylu
        :return:
        """

        if mode:
            MainScreen.background_index += 1
            if MainScreen.background_index >= len(BACKGROUNDS):
                MainScreen.background_index = 0
        else:
            MainScreen.background_index -= 1
            if MainScreen.background_index < 0:
                MainScreen.background_index = len(BACKGROUNDS)-1


class ScreenMng(ScreenManager):
    pass


class BackgroundSegmentation(App):
    def build(self) -> ScreenManager:
        Thread(target=self.webcam_output, daemon=True).start()

        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation

        self.sm = ScreenManager()
        self.main_screen = MainScreen(name="main")

        self.sm.add_widget(self.main_screen)

        return self.sm

    def webcam_output(self) -> None:
        self.do_vid = True  # flag to stop loop

        #cap = cv2.VideoCapture("video.mp4")
        cap = cv2.VideoCapture(0)
        with self.mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as sg:
            while self.do_vid:
                success, img = cap.read()

                if not success:
                    self.do_vid = False

                img = cv2.flip(img, 1)
                img = cv2.resize(img, (WIDTH, HEIGHT))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False

                results = sg.process(img)

                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.95
                bg_image = cv2.imread(BACKGROUNDS[MainScreen.background_index])

                bg_image = cv2.resize(bg_image, (WIDTH, HEIGHT))
                if MainScreen.blur:
                    bg_image = cv2.GaussianBlur(bg_image, (55, 55), 0)

                final_img = np.where(condition, img, bg_image)

                Clock.schedule_once(partial(self.show_frame, final_img))
                cv2.waitKey(1)

            cap.release()
            cv2.destroyAllWindows()

    def show_frame(self, frame, dt) -> None:
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')
        texture.flip_vertical()
        self.main_screen.ids.vid.texture = texture


if __name__ == '__main__':
    BackgroundSegmentation().run()
