import numpy as np
import win32gui
from mss import mss
from io import BytesIO

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet152

import os
import time

import icsKb as kb

import sys
from PyQt5.QtWidgets import QApplication, QWidget, \
    QPushButton, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QCoreApplication, QThread

class KartModel1(nn.Module):
  def __init__(self, class_num = 8):
    super(KartModel1, self).__init__()
    self.class_num = class_num
    self.backbone = resnet152(pretrained=True)
    in_features_num = self.backbone.fc.in_features
    
    self.backbone.fc = nn.Sequential(
      nn.Linear(in_features=in_features_num, out_features=256, bias=True),
      nn.BatchNorm1d(256),
      nn.ReLU(),
      nn.Linear(in_features=256, out_features=class_num, bias=True),
      # nn.Softmax(dim=1)
      # nn.Sigmoid(),
    )
  
  def forward(self, input_image):
    output = self.backbone(input_image)

    return output

# 기존의 main 코드와 호출하는 함수들을 묶었음
class Driver(QThread):

    def __init__(self, gui: QWidget):

        super().__init__(gui)
        self.gui = gui

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(self.device)

        self.hwnd = win32gui.FindWindow(None, "KartRider Client")
        # self.hwnd = win32gui.FindWindow(None, "카카오톡")
        if self.hwnd == 0:
            quit("Please run KartRider")
        self.rect = win32gui.GetWindowRect(self.hwnd)
        # win_pos = {"top": rect[1] + 34, "left": rect[0] + 3, "width": 1280, "height": 960}
        # 게임 클라이언트 화면 위치

        # win_pos = {"top": rect[1] + 395, "left": rect[0] + 1045, "width": 225, "height": 205}
        self.win_pos = {"top": self.rect[1] + 389, "left": self.rect[0] + 1037, "width": 223, "height": 212}
        # 미니맵 화면 위치

        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()

        self.isRunning = False

    def load_model(self):
        save_folder = "./model-weights"
        model_name = "cnn.pt"
        save_path = os.path.join(save_folder, model_name)
        model = KartModel1()
        model.load_state_dict(torch.load(save_path))
        return model
        
    def get_game_image(self, win_pos):
        sct = mss()
        sct_img = sct.grab(win_pos)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        # img.show()
        with BytesIO() as f:
            img.save(f, format="JPEG")
            f.seek(0)
            img = Image.open(f)
            img.load()
        return img

    def image_preprocessing(self, img):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        input_tensor = preprocess(img)
        return input_tensor.unsqueeze(0)
    
    def run(self):

        result_string = '100000'
        cnt = 0
        self.isRunning = True

        while self.isRunning:
            start_time = time.time()

            game_image = self.image_preprocessing(self.get_game_image(self.win_pos))
            result = self.model(game_image.to(self.device))
            pred = torch.argmax(result, dim=-1).item()

            result_string = f'{pred:03b}000'

            print(f"추론결과 : {result_string}")
            self.gui.inputLabel.setText(f"추론결과 : {result_string}")

            kb.str2keys(result_string)

            t = time.time() - start_time
            print(f"실행시간 : {t:.3}")
            self.gui.timeLabel.setText(f"실행시간 : {t:.3}")
            if t < 0.1:
                time.sleep(0.1 - t)

            cnt += 1


class MyApp(QWidget):

    def __init__(self):
        
        super().__init__()
        self.initUI()

    def initUI(self):

        self.timeLabel = QLabel("이곳에 실행시간")
        font = self.timeLabel.font()
        font.setPointSize(20)
        self.timeLabel.setFont(font)

        self.inputLabel = QLabel("이곳에 추론결과")
        font = self.inputLabel.font()
        font.setPointSize(20)
        self.inputLabel.setFont(font)
        
        startButton = QPushButton("Start")
        stopButton = QPushButton("Quit")
        
        startButton.clicked.connect(self.start)
        stopButton.clicked.connect(self.quit)
        
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(startButton)
        hbox.addWidget(stopButton)
        hbox.addStretch(1)
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.inputLabel)
        vbox.addWidget(self.timeLabel)
        vbox.addLayout(hbox)
                
        self.setLayout(vbox)
        
        self.setWindowTitle('player')
        self.move(300, 300)
        self.resize(400, 200)
        self.show()
        
        self.driver = Driver(self)
    
    def start(self):
        
        self.driver.start()

    def quit(self):

        self.driver.isRunning = False
        time.sleep(0.1)
        kb.str2keys("000000")
        QCoreApplication.instance().quit()


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        ex = MyApp()
        sys.exit(app.exec_())
    except:
        pass