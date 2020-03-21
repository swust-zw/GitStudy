#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from aip import AipOcr
import cv2
from PIL import Image
import cv2
APP_ID = '11738150'
API_KEY = '5du4OCOlph910leeymPPbeqz'
SECRET_KEY = 'riDi9UG4wgYZQ8fGZlHrPGLpjUAvsybH'
aipOcr  = AipOcr(APP_ID, API_KEY, SECRET_KEY)
# 读取图片
# filePath = "D:\img\cut_res\\c19.png"
# img=cv2.imread(filePath)
def rect_init(file):
    img = cv2.imread(file)  # 读取图像
    height, width = img.shape[:2]
    # print(img.shape)
    n, m = height // 200, width // 200
    n = min(n, m)
    if n > 1:
        img = cv2.resize(img, (width // n, height // n), interpolation=cv2.INTER_CUBIC)
    # gaus = cv2.GaussianBlur(img, (3,3),5)  # 去噪

    gaus = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # s = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    # erod = cv2.erode(gray, s)
    cv2.imwrite("img/result.png", gaus)
    cv2.imshow("gaus",gaus)
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()
#圆环印章
def circle(file):
    char=''
    img1=get_file_content(file)
# # 调用通用文字识别接口
    result = aipOcr.basicAccurate(img1)
#     result=aipOcr.basicGeneral(img1)
    char=''
    print("结果：",result)
    for word in result['words_result']:
         for w in word['words']:
            if w >= u'\u4e00' and w <= u'\u9fa5':
                char+=w
    return char
def rect(file):
    char = ''
    img1 = get_file_content(file)
    result = aipOcr.basicAccurate(img1)
    print(result)
    for word in result['words_result']:
        if word['words'] >= u'\u4e00' and word['words'] <= u'\u9fa5':
            char += word['words']
    return char
if __name__ == '__main__':
    file1='img/result.png'
    file2=r'D:\img\r-rect\sela1821.png'
    f = r"D:\img\r-rect\sela" + str(2182) + ".png"
    # rect_init(f)
    char = circle(file1)
    print(char)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # rect_init(file)