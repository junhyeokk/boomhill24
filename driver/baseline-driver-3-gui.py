import numpy as np
import win32gui
from mss import mss
import ctypes
import csv
import random
from io import BytesIO

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50

import os
import time

import icsKb as kb

import sys
from PyQt5.QtWidgets import QApplication, QWidget, \
    QPushButton, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QCoreApplication, QThread

class KartModel8(nn.Module):
    def __init__(self, num_class = 64, cnn_to_lstm = 256, lstm_hidden = 128, num_layers = 4, dropout_rate = 0.2):
        super(KartModel8, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = lstm_hidden

        self.resnet_minimap = resnet50(pretrained=True)
        self.resnet_minimap.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=cnn_to_lstm, bias=True),
            nn.ReLU(),
        )
        self.lstm_minimap = nn.LSTM(
            input_size = cnn_to_lstm,
            hidden_size = lstm_hidden,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout_rate,
        )

        self.resnet_game = resnet50(pretrained=True)
        self.resnet_game.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=cnn_to_lstm, bias=True),
            nn.ReLU(),
        )
        self.lstm_game = nn.LSTM(
            input_size = cnn_to_lstm,
            hidden_size = lstm_hidden,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout_rate,
        )

        self.lstm_features = nn.LSTM(
            input_size = 6,
            hidden_size = lstm_hidden,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout_rate,
        )

        # self.bn0 = nn.BatchNorm1d(lstm_hidden * 3)
        self.fc_1 = nn.Linear(lstm_hidden * 3, lstm_hidden * 2)
        self.bn1 = nn.BatchNorm1d(lstm_hidden * 2)
        self.relu = nn.ReLU()
        # self.dropout1 = nn.Dropout(dropout_rate)
        self.fc_2 = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.bn2 = nn.BatchNorm1d(lstm_hidden)
        self.fc_3 = nn.Linear(lstm_hidden, num_class)
        # self.sigmoid = nn.Sigmoid()

  # 전체 게임화면, 미니맵, 속도, 이전입력
    def forward(self, games, minimaps, key_inputs, hidden1 = None, hidden2 = None, hidden3 = None):
        for t in range(minimaps.size(1)):
            # with torch.no_grad():
            x1 = self.resnet_minimap(minimaps[:, t, :, :, :])
            out1, hidden1 = self.lstm_minimap(x1.unsqueeze(1), hidden1)
        # batch first = True
        # batch, seq, hidden_size

        for t in range(games.size(1)):
            x2 = self.resnet_game(games[:, t, :, :, :])
            out2, hidden2 = self.lstm_game(x2.unsqueeze(1), hidden2)

        out3, hidden3 = self.lstm_features(key_inputs, hidden3)
        # batch, seq, features

        # out = self.bn0(torch.cat([out1[:, -1, :], out2[:, -1, :], out3[:, -1, :]], dim=1))
        out = self.fc_1(torch.cat([out1[:, -1, :], out2[:, -1, :], out3[:, -1, :]], dim=1))
        # 마지막 sequence
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout1(out)
        out = self.fc_2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc_3(out)
        # out = self.sigmoid(out)

        return out, hidden1, hidden2, hidden3


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
        self.win_pos = {"top": self.rect[1] + 34, "left": self.rect[0] + 3, "width": 1280, "height": 960}
        # 게임 클라이언트 화면 위치

        # win_pos = {"top": rect[1] + 395, "left": rect[0] + 1045, "width": 225, "height": 205}
        # self.win_pos = {"top": self.rect[1] + 389, "left": self.rect[0] + 1037, "width": 223, "height": 212}
        # get_game_image(win_pos)
        # exit()
        
        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()
        self.isRunning = False

    def load_model(self):
        num_classes = 64
        save_folder = "../model/models/"
        model_name = "final_model_ep1_part1.pt"
        save_path = os.path.join(save_folder, model_name)
        model = KartModel8()
        model.load_state_dict(torch.load(save_path))
        return model
        
    def get_game_image(self, win_pos):
        sct = mss()
        sct_img = sct.grab(win_pos)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

        with BytesIO() as f:
            img.save(f, format="JPEG")
            f.seek(0)
            img = Image.open(f)
            img.load()
            
        minimap = img.crop((1034, 355, 1257, 567))

        # img.show()
        # minimap.show()
        
        return img, minimap

    def image_preprocessing(self, img, minimap):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        input_tensor = preprocess(img)
        minimap_tensor = preprocess(minimap)
        return input_tensor.unsqueeze(0).unsqueeze(0), minimap_tensor.unsqueeze(0).unsqueeze(0)
        # 배치 차원 추가 + 시간 차원 추가
    
    def run(self):
    
        hidden1, hidden2, hidden3 = None, None, None
        result_string = '000000'

        cnt = 0
        softmax = torch.nn.Softmax(dim=-1)

        self.isRunning = True

        while self.isRunning:
            start_time = time.time()

            game_image, minimap = self.image_preprocessing(*(self.get_game_image(self.win_pos)))
            past_result = torch.Tensor(list(map(int, list(result_string)))).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                result, hidden1, hidden2, hidden3 = self.model(game_image.to(self.device), minimap.to(self.device), past_result.to(self.device), hidden1, hidden2, hidden3)
            # result, hidden1, hidden2 = self.model(game_images.to(self.device), key_inputs.to(self.device))
            
            p = softmax(result)[0]
            result = torch.argmax(result, dim=-1).item()
            p = (p[result]).item()
            result_string = f'{result:06b}'
            
            print(f"추론결과 : {result_string}")
            self.gui.inputLabel.setText(f"추론결과 : {result_string}")

            kb.str2keys(result_string)

            print(f"실행시간 : {time.time() - start_time}")
            t = time.time() - start_time
            self.gui.timeLabel.setText(f"확률 : {p:.3}")
            if t < 0.1:
                time.sleep(0.1 - t)
            
            # self.isRunning = False
            cnt += 1
            # if cnt >= 15:
            #     hidden1, hidden2 = None, None
            #     cnt = 0

            # if cnt >= 100:
            #     break


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
