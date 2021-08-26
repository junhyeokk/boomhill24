import numpy as np
import win32gui
from mss import mss
import ctypes
from io import BytesIO

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, resnet34

import os
import time

import icsKb as kb

class KartModel1(nn.Module):
    def __init__(self, class_num = 8):
        super(KartModel1, self).__init__()
        self.class_num = class_num
        self.backbone = resnet34(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=class_num, bias=True),
            # nn.Softmax(dim=1)
            # nn.Sigmoid(),
        )
  
    def forward(self, input_image):
        output = self.backbone(input_image)

        return output

def load_model():
    num_classes = 64
    save_folder = "../model/models/"
    model_name = "test_model.pt"
    save_path = os.path.join(save_folder, model_name)
    # model = resnet50()
    # model.fc = nn.Sequential(
    #     nn.Linear(in_features=2048, out_features=num_classes, bias=True),
    #     # nn.Softmax(dim=1)
    # )
    model = KartModel1()
    model.load_state_dict(torch.load(save_path))
    return model

def get_game_image(win_pos):
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

def image_preprocessing(img):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(),
    ])
    input_tensor = preprocess(img)
    return input_tensor.unsqueeze(0)

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    hwnd = win32gui.FindWindow(None, "KartRider Client")
    # hwnd = win32gui.FindWindow(None, "카카오톡")
    if hwnd == 0:
        quit("Please run KartRider")
    rect = win32gui.GetWindowRect(hwnd)
    win_pos = {"top": rect[1] + 389, "left": rect[0] + 1037, "width": 223, "height": 212}

    model = load_model()
    model.to(device)
    model.eval()

    while True:
        start_time = time.time()

        game_image = image_preprocessing(get_game_image(win_pos))
        # result = torch.argmax(model(game_image.to(device)))
        result = model(game_image.to(device))
        pred = torch.argmax(result, dim=-1).item()
        # result = result.squeeze().cpu() 
        # result = result > 0.45
        # print(pred)
        result_string = f'{pred:03b}000'
        print(f"추론결과 : {result_string}")

        # result에 맞춰 키 입력
        kb.str2keys(result_string)

        t = time.time() - start_time
        # print(f"실행시간 : {t}")

        if t < 0.1:
            time.sleep(0.1 - t)
        
        print(f"실행시간 : {time.time() - start_time}")
