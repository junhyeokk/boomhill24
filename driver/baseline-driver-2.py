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

window_size = 50

class KartModel2(nn.Module):
  def __init__(self, class_num = 6):
    super(KartModel2, self).__init__()
    self.class_num = class_num
    self.backbone = resnet50(pretrained=True)
    self.backbone.fc = nn.Sequential(
      nn.Linear(in_features=2048, out_features=1000, bias=True),
      nn.ReLU(),    # 새로 추가
    )
    self.fc_include_past_inputs = nn.Sequential(
        nn.Linear(in_features=1000+6*window_size, out_features=516, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=516, out_features=256, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=128, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=class_num, bias=True),
        nn.ReLU(),
        nn.Sigmoid(),
    )
  
  def forward(self, input_image, past_inputs_10):
    out = self.backbone(input_image)
    out = self.fc_include_past_inputs(torch.cat((out, past_inputs_10), dim=1))
    return out

def load_model():
    num_classes = 64
    save_folder = "../model/models/"
    model_name = "test_model2.pt"
    save_path = os.path.join(save_folder, model_name)
    # model = resnet50()
    # model.fc = nn.Sequential(
    #     nn.Linear(in_features=2048, out_features=num_classes, bias=True),
    #     # nn.Softmax(dim=1)
    # )
    model = KartModel2()
    model.load_state_dict(torch.load(save_path))
    return model

def get_game_image(win_pos):
    sct = mss()
    sct_img = sct.grab(win_pos)
    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    # img.show()
    return img

def image_preprocessing(img):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),     # -1 ~ 1 로 normalize
    ])
    input_tensor = preprocess(img)
    # tf = transforms.ToPILImage()
    # tf(input_tensor).show()
    
    return input_tensor.unsqueeze(0)

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    hwnd = win32gui.FindWindow(None, "KartRider Client")
    # hwnd = win32gui.FindWindow(None, "카카오톡")
    if hwnd == 0:
        quit("Please run KartRider")
    rect = win32gui.GetWindowRect(hwnd)
    win_pos = {"top": rect[1] + 34, "left": rect[0] + 3, "width": 1280, "height": 960}
    # 게임 클라이언트 화면 위치

    model = load_model()
    model.to(device)
    model.eval()

    past_inputs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * window_size

    while True:
        start_time = time.time()

        game_image = image_preprocessing(get_game_image(win_pos))

        result = model(game_image.to(device), torch.Tensor(past_inputs).unsqueeze(0).to(device))
        result = result.squeeze().cpu()
        result = result > 0.5
        # result_string = f'{result:06b}'
        # print(f"추론결과 : {result}")
        if result[1] or result[2] or result[3] or result[4] or result[5]:
            print("asfd")
        
        for _ in range(6):
            del past_inputs[0]
        past_inputs += result.tolist()

        # result에 맞춰 키 입력
        kb.str2keys(result)

        # print(f"실행시간 : {time.time() - start_time}")