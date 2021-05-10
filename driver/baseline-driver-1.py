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

def load_model():
    num_classes = 64
    save_folder = "../model/models/"
    model_name = "test_model.pt"
    save_path = os.path.join(save_folder, model_name)
    model = resnet50()
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=num_classes, bias=True),
        nn.Softmax(dim=1)
    )
    model.load_state_dict(torch.load(save_path))
    return model

def get_game_image(win_pos):
    sct = mss()
    sct_img = sct.grab(win_pos)
    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
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
    hwnd = win32gui.FindWindow(None, "KartRider Client")
    # hwnd = win32gui.FindWindow(None, "카카오톡")
    if hwnd == 0:
        quit("Please run KartRider")
    rect = win32gui.GetWindowRect(hwnd)
    win_pos = {"top": rect[1] + 34, "left": rect[0] + 3, "width": 1280, "height": 960}
    # 게임 클라이언트 화면 위치

    model = load_model()
    model.eval()

    while True:
        start_time = time.time()

        game_image = image_preprocessing(get_game_image(win_pos))
        result = np.argmax(model(game_image).detach().numpy())
        result_string = f'{result:06b}'
        print(f"추론결과 : {result_string}")

        # result에 맞춰 키 입력

        print(f"실행시간 : {time.time() - start_time}")