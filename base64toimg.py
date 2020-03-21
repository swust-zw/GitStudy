#ecoding=gbk
import base64
import cv2
import numpy as np
import numpy as np
import cv2
from PIL import Image
def init_img(file):
    imq = Image.open(file)
    r,g,b,a = imq.split()
    imp = Image.new('RGB', (imq.size[0], imq.size[1]), (255, 255, 255))
    imp.paste(imq,(0, 0),mask = a)
    imp.save('img\yingzhang.png')
def img(imgdata):
    flag='base64'
    if flag in imgdata:
        imgdata=imgdata.split(flag)[1]
    img=base64.b64decode(imgdata)
    with open('img/yingzhang.png','wb') as f:
        f.write(img)
def rotate_bound_white_bg(image, angle):
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
def judgle(file):
    img = cv2.imread(file)
    height, width = img.shape[:2]
    n, m = height / 200, width / 200
    n = min(n, m)
    img = cv2.resize(img, (int(height/n), int(width/m)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 10)
    kernel = np.ones((5, 5), np.uint8)
    kernel1 = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel1)
    cv2.imshow("closing", binary)
    binary = cv2.erode(binary, kernel1, iterations=5)
    binary = rotate_bound_white_bg(binary, 45)
    img = rotate_bound_white_bg(img, 45)
    # cv2.imshow("bin0", binary)
    binary = cv2.bitwise_not(binary)
    # cv2.imshow("bin",binary)
    # cv2.imshow("opening",opening)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    a = []
    w, h = img.shape[:2]
    print(w * h)
    area_s = {}
    flag=-1
    for cnt in range(len(contours)):
        # 提取与绘制轮廓
        # img = cv2.drawContours(img, contours, cnt, (255, 0, 0), 2)
        area = cv2.contourArea(contours[cnt])  # 计算面积
        flag = 0
        # print(area)
        if area > w*h/20 and area<w*h*0.8:
            # print(area)
            area_s[cnt]=area
            # img = cv2.drawContours(img, contours, cnt, (255, 0, 0), 2)
    # print("hahah",area_s)
    max_area=max(area_s.items(),key=lambda x:x[1])
    cnt=max_area[0]
    epsilon = 0.01 * cv2.arcLength(contours[cnt], True)
    # print(epsilon)
    approx = cv2.approxPolyDP(contours[cnt], epsilon, True)
    corners = len(approx)
    print(corners)
    if corners >= 4 and corners <= 8:
        # print("**************矩形")
        flag = 1
    # if flag == 0:
    #     print("-------------圆形")
    # cv2.imshow("i", img)
    return flag
if __name__ == '__main__':
    file = "D:\\img\\seal-test2\\seal" + str(1) + ".png"
    file1 = 'img\yingzhang.png'
    init_img(file)
    judge = judgle(file1)
    # img = cv2.imread(file1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()