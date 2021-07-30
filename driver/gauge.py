import os
import numpy as np
import cv2


def load_gauge(filepath, enginetype):
    gauge = []
    numpath = os.path.join(filepath, enginetype + 'num/')
    for i in np.arange(10):
        gaugepath = os.path.join(numpath + f'/num{i}.png')
        gaugenum = cv2.imread(gaugepath, cv2.IMREAD_GRAYSCALE)
        _, num_thresh = cv2.threshold(gaugenum, 135, 255, cv2.THRESH_BINARY)
        gauge.append(num_thresh)
    return gauge


# 숫자 이미지를 예시 이미지와 매칭
def gauge_match(numimg, gauge):
    _, img_thresh = cv2.threshold(numimg, 135, 255, cv2.THRESH_BINARY)
    for i, num in enumerate(gauge):
        diff = cv2.bitwise_xor(img_thresh, num)
        # cv2.imshow('diff',diff)
        # cv2.waitKey(0)
        diffcount = cv2.countNonZero(diff)
        # 이미지 크기 변경에 따른 수치 조절 검토
        if diffcount < 25:
            return i
    return 0


def gauge_speed(gauge, enginetype, frame):
    if enginetype == 'v1':
        gauge100pos = [649, 965]
        gauge010pos = [697, 965]
        gauge001pos = [745, 965]
        # 2자리 일 경우
        gauge10pos = [673, 965]
        gauge01pos = [721, 965]
        gauge_w = 48
        gauge_h = 56
    # in game size8
    if enginetype == 'x':
        gauge100pos = [478, 698]
        gauge010pos = [500, 698]
        gauge001pos = [522, 698]
        # 2자리 일 경우
        gauge10pos = [489, 698]
        gauge01pos = [511, 698]
        gauge_w = 22
        gauge_h = 40
    gaugeimg_100 = cv2.cvtColor(frame[gauge100pos[1]:gauge100pos[1] + gauge_h,
                                gauge100pos[0]:gauge100pos[0] + gauge_w],
                                cv2.COLOR_BGR2GRAY)
    gaugeimg_010 = cv2.cvtColor(frame[gauge010pos[1]:gauge010pos[1] + gauge_h,
                                gauge010pos[0]:gauge010pos[0] + gauge_w],
                                cv2.COLOR_BGR2GRAY)
    gaugeimg_001 = cv2.cvtColor(frame[gauge001pos[1]:gauge001pos[1] + gauge_h,
                                gauge001pos[0]:gauge001pos[0] + gauge_w],
                                cv2.COLOR_BGR2GRAY)
    gaugeimg_10 = cv2.cvtColor(frame[gauge10pos[1]:gauge10pos[1] + gauge_h,
                               gauge10pos[0]:gauge10pos[0] + gauge_w],
                               cv2.COLOR_BGR2GRAY)
    gaugeimg_01 = cv2.cvtColor(frame[gauge01pos[1]:gauge01pos[1] + gauge_h,
                               gauge01pos[0]:gauge01pos[0] + gauge_w],
                               cv2.COLOR_BGR2GRAY)
    speed = 0
    speed_100 = gauge_match(gaugeimg_100, gauge)
    speed_010 = gauge_match(gaugeimg_010, gauge)
    speed_001 = gauge_match(gaugeimg_001, gauge)
    speed_10 = gauge_match(gaugeimg_10, gauge)
    speed_01 = gauge_match(gaugeimg_01, gauge)

    speed += speed_100 * 100
    speed = speed + speed_010 * 10 if speed_100 != 0 and speed_001 != 0 else speed + speed_010
    speed += speed_001
    speed += speed_10 * 10
    speed += speed_01

    # 자리가 매치하지 않으면 0으로 계산됨(무시됨)
    return speed


def boost_v1(frame: np.ndarray) -> bool:
    """
    v1 엔진의 부스터가 켜져있으면 True 꺼져있으면 False를 return함
    """
    # 특정 4x4 범위 픽셀의 HSV 평균을 구함 (해당 픽셀이 매우 밝은 주황에 가까울 경우)
    # H 값이 25 이하 (색이 빨강~주황에 가까울 때)
    # V 값이 230 이상 (밝기가 매우 밝을 때)
    hsv = cv2.cvtColor(frame[743:747, 425:429], cv2.COLOR_BGR2HSV)
    return True if np.average(hsv[:, :, 2]) > 230 and np.average(hsv[:, :, 0]) < 25 else False
