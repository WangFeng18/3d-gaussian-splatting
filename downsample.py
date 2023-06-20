import glob
import os
import cv2

for i, img_path in enumerate(glob.glob("images_2/*.JPG")):
    img = cv2.imread(img_path)
    w, h = img.shape[1], img.shape[0]
    img = cv2.resize(img, (w//2, h//2))
    cv2.imwrite("images_4/{}".format(os.path.basename(img_path)), img)
    print(i)
