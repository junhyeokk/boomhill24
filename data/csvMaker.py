# ref: https://blog.naver.com/dlwjdskfcl/222403253767

import ctypes
import time

import pandas as pd

import win32gui
from mss import mss
from PIL import Image

IMG_PATH = "../img/"      # 스샷 저장 경로
FILENAME = "../test.csv"  # csv 파일경로 및 이름

INTERVAL = 0.1

ENTER = 0x0D
HANJA = 0x19
SPACE = 0x20

LEFT   = 0x25        # LEFT ARROW key
UP     = 0x26          # UP ARROW key
RIGHT  = 0x27       # RIGHT ARROW key
DOWN   = 0x28        # DOWN ARROW key
LSHIFT = 0xA0
LCTRL  = 0xA2

keymap = [UP, LEFT, RIGHT, LCTRL, LSHIFT, DOWN]


def getkey(vkKeyCode):
    return ctypes.windll.user32.GetAsyncKeyState(vkKeyCode) > 1


if __name__ == '__main__':

    hwnd = win32gui.FindWindow(None, "KartRider Client")
    if hwnd == 0:
        quit("Please run KartRider")
    rect = win32gui.GetWindowRect(hwnd)
    win_pos = {"top": rect[1] + 311, "left": rect[0] + 828, "width": 180, "height": 169}

    print("Press ENTER to record...")
    while True:
        if getkey(ENTER):
            break
        time.sleep(INTERVAL)

    print("START!!")
    print("Press SPACEBAR to stop...")

    imgnames = []
    inputs = []
    cnt = 0

    while True:
        
        start_time = time.time()

        if getkey(SPACE):
            break

        # 이미지 저장하며 플레이
        sct = mss()
        sct_img = sct.grab(win_pos)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        imgfile = str(cnt) + ".jpg"
        imgnames.append(imgfile)
        img.save(IMG_PATH + imgfile)

        # 키 입력상태 읽어서 문자열로 생성
        keystr = ""
        for idx in range(len(keymap)):
            if getkey(keymap[idx]):
                keystr = keystr + '1'
            else:
                keystr = keystr + '0'
        inputs.append(keystr)

        # print(imgname)
        print(keystr)

        t = time.time() - start_time
        if t < 0.1:
            time.sleep(INTERVAL - t)

        cnt += 1
    
    df = pd.DataFrame({'imgname': imgnames, 'input': inputs})
    df.to_csv(FILENAME, header = False, index = False)