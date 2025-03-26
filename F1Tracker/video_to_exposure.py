import cv2
import numpy as np


path = r"C:\Users\joshu\Documents\Figs\Levitation\F1_Track.mp4"
vidcap = cv2.VideoCapture(path)
success,image = vidcap.read()
print(success)
print(image.shape)

images = []

count = 0
while success:
    success,image = vidcap.read()
    if success:
        images.append(image.copy())
    # print('Read a new frame: ', success)

a = 0
imgs = np.zeros_like(images[0])
for img in images:
    a += img[0,0,0]
    imgs += img
print(a)
print(img.shape)
img  = sum(images, start=np.zeros_like(images[0]))
# print(img)
cv2.imwrite("frame.png", img )
