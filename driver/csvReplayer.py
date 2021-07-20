# ref: https://wikidocs.net/book/2165

# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QApplication, QWidget, \
    QPushButton, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QCoreApplication, QThread

import directkeys as kb
import pandas as pd
import time
import os

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50

# 키 누르는 간격 (단위: 초)
INTERVAL = 0.1

# csv 파일 경로
CSVPATH = "./kart_test_minimap_test_0.csv"

keymap = [kb.UP, kb.LEFT, kb.RIGHT, kb.LCTRL, kb.LSHIFT, kb.DOWN]


class KartModel8(nn.Module):
  def __init__(self, num_class = 64, cnn_to_lstm = 1024, lstm_hidden = 512, num_layers = 5):
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
        input_size = cnn_to_lstm,
        hidden_size = lstm_hidden,
        num_layers = num_layers,
        batch_first = True,
        dropout = 0.3,
    )
    self.lstm_key = nn.LSTM(
        input_size = 6,
        hidden_size = lstm_hidden,
        num_layers = num_layers,
        batch_first = True,
        dropout = 0.3,
    )

    self.fc_1 = nn.Linear(lstm_hidden * 2, lstm_hidden * 2)
    self.bn1 = nn.BatchNorm1d(lstm_hidden * 2)
    self.relu = nn.ReLU()
    self.fc_2 = nn.Linear(lstm_hidden * 2, lstm_hidden)
    self.bn2 = nn.BatchNorm1d(lstm_hidden)
    self.fc_3 = nn.Linear(lstm_hidden, num_class)
    # self.sigmoid = nn.Sigmoid()

  def forward(self, x_3d, key_inputs, hidden1 = None, hidden2 = None):
    for t in range(x_3d.size(1)):
      with torch.no_grad():
        x = self.resnet(x_3d[:, t, :, :, :])
        out1, hidden1 = self.lstm_image(x.unsqueeze(1), hidden1)
    # batch first = True
    # batch, seq, hidden_size

    out2, hidden2 = self.lstm_key(key_inputs, hidden2)

    out = self.fc_1(torch.cat([out1[:, -1, :], out2[:, -1, :]], dim=1))
    # 마지막 sequence
    out = self.bn1(out)
    out = self.relu(out)
    out = self.fc_2(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.fc_3(out)
    # out = self.sigmoid(out)

    return out, hidden1, hidden2


def str2keys(result):
    
    for idx in range(6):
        if '1' == result[idx]:
            kb.PressKey(keymap[idx])
        else:
            kb.ReleaseKey(keymap[idx])


class CsvReplayThread(QThread):

    def __init__(self, gui: QWidget, csvpath: str):

        super().__init__(gui)
        self.gui = gui
        self.csvpath = csvpath

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(self.device)

        # self.hwnd = win32gui.FindWindow(None, "KartRider Client")
        # self.hwnd = win32gui.FindWindow(None, "카카오톡")
        # if self.hwnd == 0:
        #     quit("Please run KartRider")
        # self.rect = win32gui.GetWindowRect(self.hwnd)
        # win_pos = {"top": rect[1] + 34, "left": rect[0] + 3, "width": 1280, "height": 960}
        # 게임 클라이언트 화면 위치

        # win_pos = {"top": rect[1] + 395, "left": rect[0] + 1045, "width": 225, "height": 205}
        # self.win_pos = {"top": self.rect[1] + 389, "left": self.rect[0] + 1037, "width": 223, "height": 212}
        # get_game_image(win_pos)
        # exit()

        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()

        self.isRunning = False

    # def __init__(self, parent: QWidget, csvpath: str):

    #     super().__init__(parent)
    #     self.parent = parent
    #     self.csvpath = csvpath
    #     self.isRunning = False

    def load_model(self):
        num_classes = 64
        save_folder = "../model/models/"
        model_name = "test_model_minimap5.pt"
        save_path = os.path.join(save_folder, model_name)
        model = KartModel8()
        model.load_state_dict(torch.load(save_path))
        return model
        
    # def get_game_image(self, win_pos):
    #     sct = mss()
    #     sct_img = sct.grab(win_pos)
    #     img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    #     # img.show()
    #     return img

    def image_preprocessing(self, img):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        input_tensor = preprocess(img)
        return input_tensor.unsqueeze(0)
        # 배치 차원 추가
    
    def run(self):

        hidden1 = None
        hidden2 = None
        game_image_list = []
        key_input_list = []
        result_string = '100000'
        cnt = 0

        self.isRunning = True

        df = pd.read_csv(self.csvpath, names=["filename", "keyinput"], dtype={"filename": str, "keyinput": str})
        corr_rate = 0
        corr_total = 0
        err_rate = 0
        err_total = 0
        for row in df.itertuples():
            if cnt == 0:
                cnt += 1
                continue

            filename, keyinput = row[1], row[2]
            game_image = self.image_preprocessing(Image.open("./" + filename))
            # str2keys(keyinput)

            # many-to-one 방식 추론
            # game_image_list.append(game_image)
            # if len(game_image_list) > 10:
            #     game_image_list.pop(0)
            # game_images = torch.stack(game_image_list, dim = 1)

            # past_result = torch.Tensor(list(map(int, list(result_string)))).unsqueeze(0)
            # key_input_list.append(past_result)
            # if len(key_input_list) > 10:
            #     key_input_list.pop(0)
            # key_inputs = torch.stack(key_input_list, dim = 1)

            # result, hidden1, hidden2 = self.model(game_images.to(self.device), key_inputs.to(self.device))

            # many-to-many 방식 추론
            game_images = game_image.unsqueeze(0)
            past_result = torch.Tensor(list(map(int, list(result_string)))).unsqueeze(0).unsqueeze(0)
            result, hidden1, hidden2 = self.model(game_images.to(self.device), past_result.to(self.device), hidden1, hidden2)

            softmax = torch.nn.Softmax(dim=-1)
            p = softmax(result)[0]
            result = torch.argmax(result, dim=-1).item()
            result_string = f'{result:06b}'
            
            print(f"추론결과 : {result_string}")
            print(f"정답 : {keyinput}")
            self.gui.inputLabel.setText(f"추론결과 : {result_string}")

            self.gui.inputLabel.setText(keyinput)
            if not self.isRunning:
                break
            # time.sleep(INTERVAL)
            
            p = (p[result]).item()
            if (result_string == keyinput):
                corr_rate += p
                corr_total += 1
                print(f"correct : {p:.3}")
            else:
                err_rate += p
                err_total += 1

            result_string = keyinput
        print(corr_rate / corr_total)
        print(err_rate / err_total)
        print((corr_total) / (corr_total + err_total))


class MyApp(QWidget):

    def __init__(self):
        
        super().__init__()
        self.initUI()

    def initUI(self):
        
        self.inputLabel = QLabel("waiting...")
        font = self.inputLabel.font()
        font.setPointSize(20)
        self.inputLabel.setFont(font)
        
        startButton = QPushButton("Start")
        stopButton = QPushButton("Quit")
        
        startButton.clicked.connect(self.start)
        stopButton.clicked.connect(QCoreApplication.instance().quit)
        
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(startButton)
        hbox.addWidget(stopButton)
        hbox.addStretch(1)
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.inputLabel)
        vbox.addLayout(hbox)
                
        self.setLayout(vbox)
        
        self.setWindowTitle('csv replayer')
        self.move(300, 300)
        self.resize(400, 200)
        self.show()
    
    def start(self):
        
        self.t = CsvReplayThread(self, CSVPATH)
        self.t.start()


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        ex = MyApp()
        sys.exit(app.exec_())
    except:
        pass