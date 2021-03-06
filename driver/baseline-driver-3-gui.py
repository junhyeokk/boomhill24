import numpy as np
import win32gui
from mss import mss
import ctypes
import cv2

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

import gauge


class KartModel8(nn.Module):
    def __init__(self, num_class=64, cnn_to_lstm=1024, lstm_hidden=512, num_layers=5):
        super(KartModel8, self).__init__()
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = lstm_hidden

        self.resnet = resnet50(pretrained=False)
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=cnn_to_lstm, bias=True),
            nn.ReLU(),
        )
        self.lstm_image = nn.LSTM(
            input_size=cnn_to_lstm,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )
        self.lstm_key = nn.LSTM(
            input_size=6,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
        )

        self.fc_1 = nn.Linear(lstm_hidden * 2, lstm_hidden * 2)
        self.bn1 = nn.BatchNorm1d(lstm_hidden * 2)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.bn2 = nn.BatchNorm1d(lstm_hidden)
        self.fc_3 = nn.Linear(lstm_hidden, num_class)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x_3d, key_inputs, hidden1=None, hidden2=None):
        for t in range(x_3d.size(1)):
            #   with torch.no_grad():
            x = self.resnet(x_3d[:, t, :, :, :])
            out1, hidden1 = self.lstm_image(x.unsqueeze(1), hidden1)
        # batch first = True
        # batch, seq, hidden_size

        out2, hidden2 = self.lstm_key(key_inputs, hidden2)

        out = self.fc_1(torch.cat([out1[:, -1, :], out2[:, -1, :]], dim=1))
        # ????????? sequence
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc_3(out)
        # out = self.sigmoid(out)

        return out, hidden1, hidden2


# ????????? main ????????? ???????????? ???????????? ?????????
class Driver(QThread):

    def __init__(self, gui: QWidget):

        super().__init__(gui)
        self.gui = gui

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(self.device)

        self.hwnd = win32gui.FindWindow(None, "KartRider Client")
        # self.hwnd = win32gui.FindWindow(None, "????????????")
        if self.hwnd == 0:
            quit("Please run KartRider")
        self.rect = win32gui.GetWindowRect(self.hwnd)
        # ?????? ??????????????? ?????? ??????
        self.win_pos = {"top": self.rect[1] + 26, "left": self.rect[0] + 3, "width": 1024, "height": 768}

        # ????????? ??????
        # self.win_pos = {"top": self.rect[1] + 698 + 26, "left": self.rect[0] + 478 + 3, "width": 66, "height": 40}

        # win_pos = {"top": rect[1] + 395, "left": rect[0] + 1045, "width": 225, "height": 205}
        # self.win_pos = {"top": self.rect[1] + 389, "left": self.rect[0] + 1037, "width": 223, "height": 212}
        # get_game_image(win_pos)
        # exit()

        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()
        self.gauge = gauge.load_gauge('.', 'x')

        self.isRunning = False

    def load_model(self):
        num_classes = 64
        save_folder = "../model/models/"
        model_name = "test_model3.pt"
        save_path = os.path.join(save_folder, model_name)
        model = KartModel8()
        model.load_state_dict(torch.load(save_path))
        return model

    def get_game_image(self, win_pos):
        sct = mss()
        sct_img = sct.grab(win_pos)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        # img.show()
        return img

    def image_preprocessing(self, img):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        input_tensor = preprocess(img)
        return input_tensor.unsqueeze(0)
        # ?????? ?????? ??????

    def run(self):

        hidden1 = None
        hidden2 = None
        result_string = '100000'

        game_image_list = []
        key_input_list = []
        cnt = 0

        self.isRunning = True

        while self.isRunning:
            start_time = time.time()

            # ?????? ?????????
            game_image_cv2 = cv2.cvtColor(np.array(self.get_game_image(self.win_pos)), cv2.COLOR_RGB2BGR)

            # ?????? ????????? ??????
            game_speed = gauge.gauge_speed(self.gauge, 'x', game_image_cv2)
            game_boost = gauge.boost_v1(game_image_cv2)
            game_boost_per = gauge.boost_range(game_image_cv2)
            print('?????? : ', game_speed)
            print('?????? ????????? : ', game_boost)
            print("?????? ????????? ????????? : ",game_boost_per)


            # game_image = self.image_preprocessing(self.get_game_image(self.win_pos)).unsqueeze(0)
            game_image = self.image_preprocessing(self.get_game_image(self.win_pos))
            # print(game_image.shape)
            # tf = transforms.ToPILImage()
            # tf(game_image[0]).show()
            # break

            # ?????? ???????????? ?????? ?????????, ???????????? 1

            # ????????? ???????????? 10 ????????? ?????????
            game_image_list.append(game_image)
            if len(game_image_list) > 10:
                game_image_list.pop(0)
            game_images = torch.stack(game_image_list, dim=1)
            # ??????????????? ????????? ??????

            # past_result = torch.Tensor(list(map(int, list(result_string)))).unsqueeze(0).unsqueeze(0)
            past_result = torch.Tensor(list(map(int, list(result_string)))).unsqueeze(0)
            # ?????? ?????? ??????

            key_input_list.append(past_result)
            if len(key_input_list) > 10:
                key_input_list.pop(0)
            key_inputs = torch.stack(key_input_list, dim=1)
            # ??????????????? ?????? ?????? ??????

            # result, hidden1, hidden2 = self.model(game_image.to(self.device), past_result.to(self.device), hidden1, hidden2)
            result, hidden1, hidden2 = self.model(game_images.to(self.device), key_inputs.to(self.device))

            result = torch.argmax(result, dim=-1).item()
            result_string = f'{result:06b}'

            print(f"???????????? : {result_string}")
            self.gui.inputLabel.setText(f"???????????? : {result_string}")

            if result_string == '000000':
                result_string = '100000'

            # result??? ?????? ??? ??????
            kb.str2keys(result_string)

            print(f"???????????? : {time.time() - start_time}")
            t = time.time() - start_time
            self.gui.timeLabel.setText(f"???????????? : {t}")
            if t < 0.1:
                time.sleep(0.1 - t)
            # break
            cnt += 1
            if cnt >= 15:
                hidden1, hidden2 = None, None
                cnt = 0


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.timeLabel = QLabel("????????? ????????????")
        font = self.timeLabel.font()
        font.setPointSize(20)
        self.timeLabel.setFont(font)

        self.inputLabel = QLabel("????????? ????????????")
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
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
