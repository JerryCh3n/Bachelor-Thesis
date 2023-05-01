import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def on_trackbar(val):
    _, BW = cv2.threshold(255 - gray, 255 - val, 255, cv2.THRESH_BINARY)
    k1 = np.ones((21, 21), np.uint8)
    BW = cv2.morphologyEx(BW, cv2.MORPH_CLOSE, k1)

    canny = cv2.Canny(BW, 20, 35)

    ycoords = []
    xcoords = []
    for col in range(canny.shape[1]):
        whites = np.argwhere(canny[:, col])
        if whites.size >= 2:
            xcoords.append(col)
            ycoords.append(((whites.max()+whites.min())/2).astype(int))

    m, b = np.polyfit(xcoords, ycoords, 1)
    xcoords = np.array(xcoords)
    line_ycoords = m*xcoords+b

    cut_canny = cut_img.copy()
    cut_canny[canny == 255] = [0, 0, 255]

    cv2.line(cut_canny, (int(xcoords[0]), int(line_ycoords[0])), (int(
        xcoords[-1]), int(line_ycoords[-1])), (102, 204, 0), 2)

    angle = np.arctan(m)
    length = canny.shape[1]/np.cos(angle)
    width = np.round(np.sum(BW == 255) / length, 2)

    text = "W:{},T:{}".format(width, val)
    positon = (20, int(cut_canny.shape[0]-20))
    text_image = cv2.putText(
        img=cut_canny,
        text=text,
        org=positon,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 0, 0),
        thickness=3
    )
    cBW = cv2.cvtColor(BW, cv2.COLOR_GRAY2BGR)
    final_image = np.vstack([text_image, cBW])

    cv2.imshow(img_path, final_image)
    return width, final_image


# -----Start of Script-----
# This script measures the line width of images taken and exports the measurement to a csv file
# A copy of the processed image is also saved in "Steps" directory 
# The script folder navigation structure is as follows
"""
date/
|- LDR_xxx/
| |- widthxxx.jpg
| |- Steps/
| | |- processed_widthxxx.jpg
"""

date = "Nov_30"
thres_val = 75
file = open(date+".csv", "a")

subfolders = next(os.walk("./"+date))[1]
subfolders.sort()

for folder in subfolders:

    files = next(os.walk("./{}/{}".format(date, folder)))[2]
    files = [i for i in files if ".jpg" in i]
    files = [i for i in files if "width" in i]
    files.sort()

    os.makedirs("./{}/{}/Steps".format(date, folder), exist_ok=True)

    line_entry = str(folder)
    for img_name in files:
        while True:
            img_path = "./{}/{}/{}".format(date, folder, img_name)

            img = cv2.imread(img_path)
            roi = cv2.selectROI(windowName="roi", img=img,
                                showCrosshair=False, fromCenter=False)
            x, y, w, h = roi
            cut_img = img[y:y+h, x:x+w]
            gray = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
            cv2.destroyAllWindows()

            cv2.namedWindow(img_path, cv2.WINDOW_NORMAL)
            cv2.createTrackbar("Threshod", img_path,
                               thres_val, 100, on_trackbar)
            on_trackbar(thres_val)
            k = cv2.waitKey(0)

            if chr(k) == 'b':
                pass
            elif chr(k) == 'e':
                file.write(line_entry+"\n")
                file.close()
                exit()
            else:
                break

        thres_val = cv2.getTrackbarPos("Threshod", img_path)
        width, proccess_img = on_trackbar(thres_val)

        cv2.imwrite("./{}/{}/Steps/processed_{}".format(date,
                    folder, img_name), proccess_img)
        cv2.destroyAllWindows()

        line_entry += ",{},{}".format(width, thres_val)
    file.write(line_entry+"\n")
file.close()
