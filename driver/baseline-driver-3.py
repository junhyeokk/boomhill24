import numpy as np
import win32gui
from mss import mss
import ctypes

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50

import os
import time

import icsKb as kb

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

def load_model():
    num_classes = 64
    save_folder = "../model/models/"
    model_name = "test_model_minimap4.pt"
    save_path = os.path.join(save_folder, model_name)
    # model = resnet50()
    # model.fc = nn.Sequential(
    #     nn.Linear(in_features=2048, out_features=num_classes, bias=True),
    #     # nn.Softmax(dim=1)
    # )
    model = KartModel8()
    model.load_state_dict(torch.load(save_path))
    return model

def get_game_image(win_pos):
    sct = mss()
    sct_img = sct.grab(win_pos)
    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    # img.show()
    return img
86
def image_preprocessing(img):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    input_tensor = preprocess(img)
    return input_tensor.unsqueeze(0)
    # 배치 차원 추가

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    hwnd = win32gui.FindWindow(None, "KartRider Client")
    # hwnd = win32gui.FindWindow(None, "카카오톡")
    if hwnd == 0:
        quit("Please run KartRider")
    rect = win32gui.GetWindowRect(hwnd)
    # win_pos = {"top": rect[1] + 34, "left": rect[0] + 3, "width": 1280, "height": 960}
    # 게임 클라이언트 화면 위치

    # win_pos = {"top": rect[1] + 395, "left": rect[0] + 1045, "width": 225, "height": 205}
    win_pos = {"top": rect[1] + 389, "left": rect[0] + 1037, "width": 223, "height": 212}
    # get_game_image(win_pos)
    # exit()

    model = load_model()
    model.to(device)
    model.eval()

    hidden1 = None
    hidden2 = None
    result_string = '100000'

    game_image_list = []
    key_input_list = []
    cnt = 0
    while True:
        start_time = time.time()
        # game_image = image_preprocessing(Image.open('./test.jpg'))

        game_image = image_preprocessing(get_game_image(win_pos)).unsqueeze(0)
        # 게임 이미지의 배치 사이즈, 시퀸스가 1
        # 여기서 시퀸스를 10 정도로 늘리기
        # game_image_list.append(game_image)
        # if len(game_image_list) > 10:
        #     game_image_list.pop(0)
        # game_images = torch.stack(game_image_list, dim = 1)
        # 시간축으로 이미지 쌓기

        past_result = torch.Tensor(list(map(int, list(result_string)))).unsqueeze(0).unsqueeze(0)
        # 배치 차원 추가
        # key_input_list.append(past_result)
        # if len(key_input_list) > 10:
        #     key_input_list.pop(0)
        # key_inputs = torch.stack(key_input_list, dim = 1)
        # 시간축으로 이전 입력 쌓기

        result, hidden1, hidden2 = model(game_image.to(device), past_result.to(device), hidden1, hidden2)

        result = torch.argmax(result, dim=-1).item()
        result_string = f'{result:06b}'
        
        print(f"추론결과 : {result_string}")
        # result에 맞춰 키 입력
        kb.str2keys(result_string)

        # if result_string == '000000':
        #     result_string = '100000'

        print(f"실행시간 : {time.time() - start_time}")
        # break
        # cnt += 1
        # if cnt == 30:
        #     break