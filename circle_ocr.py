# coding=gbk
import cv2
import numpy as np
import math
import random
from sklearn.cluster import KMeans

def init(file):
    img = cv2.imread(file)  # 读取图像
    height, width = img.shape[:2]
    # print(img.shape)
    n, m = height // 200, width // 200
    n = min(n, m)
    if n > 1:
        img = cv2.resize(img, (width // n, height // n), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
    return gray

# 圆转直角函数，其中theta为顺时针
def polar(I, center, r, thetastep, theta, rstep=0.5):
    minr, maxr = r
    h = I.shape[0]
    w = I.shape[1]
    cx, cy = center
    mintheta, maxtheta = theta
    H = int((maxr - minr) / rstep) + 1
    W = int((maxtheta - mintheta) / thetastep) + 1
    O = 125 * np.ones((H, W), I.dtype)
    r = np.linspace(minr, maxr, H)
    r = np.tile(r, (W, 1))
    r = np.transpose(r)
    theta = np.linspace(mintheta, maxtheta, W)
    theta = np.tile(theta, (H, 1))
    x, y = cv2.polarToCart(r, theta, angleInDegrees=True)
    for i in range(H):
        for j in range(W):
            px = int(round(x[i][j] + cx))
            py = int(round(y[i][j] + cy))
            if (px >= 0 and px <= w - 1) and (py >= 0 and py <= h - 1):
                O[i][j] = I[py][px]
    return O
# 垂直像素值统计
def hProject(binary):
    # 水平方向投影
    h, w = binary.shape  # 把输入图像的高度与宽度赋值给h,w

    hprojection = np.zeros(binary.shape, dtype=np.uint8)  # 输入图片大小全部0的列表
    hprojection.fill(255)  # 把0全部变成255

    h_h = [0] * h  # 创建h长度都为0的数组
    # print(len(h_h))
    for j in range(h):  # 遍历每一行
        for i in range(w):  # 遍历每一行的全部列
            if binary[j, i] == 0:  # 传入的是二值化以后的图品，判断是否有字
                h_h[j] += 1
    # 画出投影图，集中显示在开始行开始部分
    for j in range(h):
        for i in range(h_h[j]):
            hprojection[j, i] = 0

    # cv2.imshow('hpro', hprojection)
    return h_h
# 数据分类
def kmeans(w_w, x1, x2):
    p1 = []
    p2 = []
    for i in w_w:
        if abs(i - x1) <= abs(i - x2):
            p1.append(i)
        else:
            p2.append(i)
    return p1, p2

def twoTypeData(test1):
    x1, x2 = min(test1), max(test1)
    for i in range(0, 20):
        p1, p2 = kmeans(test1, x1, x2)
        x1 = sum(p1) // (len(p1) + 0.1)
        x2 = sum(p2) // (len(p2) + 0.1)
    if x1 > x2:
        x1 = x2  # x1确定数字分割长度
        p1, p2 = p2, p1
    return p1, p2

# 水平像素值统计
def vProject(binary):
    # 垂直投影
    h, w = binary.shape

    # 创建 w 长度都为0的数组
    w_w = [0] * w
    for i in range(w):
        for j in range(h):
            if binary[j, i] == 0:
                w_w[i] += 1

    return w_w

def h_img(O_polar):
    h_h = hProject(O_polar)
    h, w = O_polar.shape[:2]
    h_start = []
    h_end = []
    h_min = min(h_h[:35])
    start = 0
    for i in range(len(h_h)):
        if h_h[i] >= h_min and start == 0:
            # print("star:", h_h[i], i)
            h_start.append(i)
            start = 1
        if h_h[i] <= h_min and start == 1:
            # print("end:", h_h[i], i)
            h_end.append(i)
            start = 0
        if i == len(h_h) - 1 and len(h_end) != len(h_start) and start == 1:
            h_end.append(i)
    # ----------------------------------------------------根据水平投影获取分割
    for i in range(len(h_start)):
        if h_end[i] - h_start[i] > h / 2:
            O_polar = O_polar[h_start[i]:h_end[i], ]
            break
    print("star-end:", h_start, h_end)
    return O_polar
# 二值化
def binary(img):
    img = cv2.GaussianBlur(img, (9, 9), 0)  # 去噪
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img
def split(file):
    gray = init(file)
    # 霍夫检测圆找到圆心以及半径
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200,
                               param1=100, param2=30,
                               minRadius=(gray.shape[0]) // 10,
                               maxRadius=(gray.shape[0]) // 2)
    # print(circles)
    x = int(circles[0][0][0])
    y = int(circles[0][0][1])
    r = int(circles[0][0][2])
    # ---------------------------------------------裁剪内切圆
    img_tailor = gray[abs(y - r):y + r, abs(x - r):x + r]
    h, w = img_tailor.shape[:2]
    # ---------------------------------------------极坐标转笛卡尔坐标
    cx, cy, r = int(h / 2), int(h / 2), int(h / 2)  # 圆心坐标
    O_polar = polar(img_tailor, (cx, cy), (r / 2, r), 360 / h * 0.2, (0, 360))
    O_polar = cv2.flip(O_polar, 0)
    # ----------------------------------------------二值化处理
    O_polar = binary(O_polar)

    # ----------------------------------------------------获取垂直投影值
    w_w = vProject(O_polar)
    # --------------------------------------------根据垂直将数据分为两类分割查找分割点
    p1, p2 = twoTypeData(w_w)
    cut_point = 0
    for i in range(len(w_w) // 20, len(w_w) - len(w_w) // 20):
        num = 0
        a = w_w[i - len(w_w) // 20:i + len(w_w) // 20]
        for j in a:
            if j in p1:
                num += 1
        if num / len(a) > 0.95:
            cut_point = i
            break
    if cut_point==0:
        # img_tailor[h*2//3:,w//4:w*3//4]=255
        triangle = np.array([[w//4,h], [w*3//4,h], [w*3//4,h*2//3],[w//4,h*2//3]])
        cv2.fillConvexPoly(img_tailor, np.int32([triangle]), 255)
        O_polar = polar(img_tailor, (cx, cy), (r / 2, r), 360 / h * 0.2, (90, 360+90))
        O_polar = cv2.flip(O_polar, 0)
        # cv2.imshow("crop",img_tailor)
    # flag_row = 0
    # if cut_point == 0:
    #     # 腐蚀
    #     s = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    #     erod = cv2.erode(O_polar, s)
    #     w_w1 = vProject(erod)
    #     i,w_flag= 0,0
    #     w_satr = []
    #     w_end = []
    #     while i < len(w_w1):
    #         if w_w1[i] >= (max(w_w1) + min(w_w1)) / 2 and w_flag == 0:
    #             w_satr.append(i)
    #             w_flag = 1
    #         if w_w1[i] < (max(w_w1) + min(w_w1)) / 2 and w_flag == 1:
    #             w_end.append(i)
    #             w_flag = 0
    #         if i == len(w_w1) and len(w_end) != len(w_satr):
    #             w_end.append(i)
    #         i += 1
    #     w_sub = [(w_end[i] - w_satr[i]) for i in range(0, len(w_end))]
    #     # print("减", w_sub, w_end, w_satr)
    #     j = w_sub.index(max(w_sub))
    #     w_s = w_satr[j]
    #     w_e = w_end[j]
    #     cut_point = w_s
    #     flag_row = 1
    if cut_point != 0:
        degree = int((360 / O_polar.shape[1]) * cut_point)
        O_polar = polar(img_tailor, (cx, cy), (r / 2, r), 360 / h * 0.2, (degree, degree + 360))
        O_polar = cv2.flip(O_polar, 0)
        # O_polar[0:, 0:len(w_w) // 7] = 255
        # if flag_row == 1:
        #     O_polar[0:, 0:w_e - w_s] = 255
        # cv2.imshow("r",O_polar)
    binary_img=binary(O_polar)
    # kernel1 = np.ones((2, 2), np.uint8)
    # O_polar = cv2.erode(O_polar, kernel1, iterations=1)

    # O_polar = cv2.Laplacian(O_polar, cv2.CV_64F).var()

    midle_poin=int(O_polar.shape[0]*0.5)
    len_s=0
    for i in range(O_polar.shape[1]//2):
        if binary_img[midle_poin][i]==255:
            len_s += 1
            continue
        elif binary_img[midle_poin][i]==0:
            break
    len_e=0
    for i in range(O_polar.shape[1]-1,O_polar.shape[1]//2,-1):
        if binary_img[midle_poin][i] == 255:
            len_e += 1
            continue
        elif binary_img[midle_poin][i] == 0:
            break
    if len_e>0 and len_s>15:
        O_polar=O_polar[0:,len_s-15:O_polar.shape[1]-len_e+10]
    # cv2.imshow("du",O_polar)
    cv2.imwrite("img/result.png",O_polar)
    return O_polar

if __name__ == '__main__':
    file=r'D:\PYTHON\app_ocr\img\yingzhang.png'
    test=r"D:\img\r-circle\sela1.png"
    split(test)
    # cv2.imshow("i",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


