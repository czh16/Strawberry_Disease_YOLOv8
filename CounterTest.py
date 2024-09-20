# encoding:utf-8
import cv2
import numpy as np

img = cv2.imread('UIProgram/ui_imgs/11.png', cv2.IMREAD_GRAYSCALE)
ret, thresh = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(contours)
# 复制原图
# img1 = img.copy()
# # 创建一幅相同大小的白色图像
# img2 = np.ones(img.shape)
# # 按照面积将所有轮廓逆序排序
# contours2 = sorted(contours, key=lambda a: cv2.contourArea(a), reverse=True)
# for c in contours2:
#     area = cv2.contourArea(c)
#     print(area)
#     # 只输出面积大于500轮廓
#     if area<500:break
#     # 分别在复制的图像上和白色图像上绘制当前轮廓
#     cv2.drawContours(img1, [c],0, (0,255,0), 3)
#     cv2.drawContours(img2, [c],0, (0,255,0), 3)