from app_ocr import base64toimg,circle_ocr,ocr_baidu
import cv2
file="D:\\img\\seal-test2\\"
rect=[]
circle=[]
for i in range(1,2184):
    try:
        print("******"+str(i)+"*******",end=' ')
        file="D:\\img\\seal-test2\\seal"+str(i)+".png"
        file1='img\yingzhang.png'
        base64toimg.init_img(file)
        judge = base64toimg.judgle(file1)
        img=cv2.imread(file1)
        if judge==1:
            # f="D:\\img\\r-rect\\sela"+str(i)+".png"
            # cv2.imwrite(f,img)
            rect.append(i)
        elif judge==0:
            # f = "D:\\img\\r-circle\\sela" + str(i) + ".png"
            # cv2.imwrite(f,img)
            circle.append(i)
        else:
            f = "D:\\img\\r-other\\sela" + str(i) + ".png"
            cv2.imwrite(f,img)

    except:
        print("????????????????????" + str(i) + "????????????????????")
print(rect)