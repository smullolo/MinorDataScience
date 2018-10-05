import cv2
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def gcd(x, y): # Calculate greatest common devider.
    while y != 0:
        (x, y) = (y, x % y)
    return x


def video_capture(videolink, frameinterval):
    vidcap = cv2.VideoCapture(videolink)  # load video
    success, image = vidcap.read()  # check if video is read succesfully
    count = 0  # variable to count the frames.
    framecap = frameinterval  # set interval for saving frames.
    name, extension = os.path.splitext(os.path.basename(videolink)) # split filename in file and extension.
    width = 720
    height = 510
    upper_border = 30

    while success:
        if count % framecap == 0:  # check if the framecount can be devided by 300
            cropped = image[upper_border:height, 0:width]
            # gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("%s_frame%d.jpg" % (name, count / framecap), cropped)  # save image every 300th frame

        success, image = vidcap.read()
        print("read a new frame succes:", success)
        count += 1


def split_image(image_source, height, width): # load image and split it into squares.
    image = cv2.imread(image_source)
    name, extension = os.path.splitext(os.path.basename(image_source))
    divider = gcd(height, width)
    y_length = int(height / divider)
    x_length = int(width / divider)
    print(y_length)
    print(x_length)
    for x in range(x_length):
        for y in range(y_length):
            y_start = y * divider
            x_start = x * divider
            segment = image[y_start:(y_start + divider), x_start:(x_start + divider)]
            cv2.imwrite("%s_segment(%d,%d).jpg" % (name, x, y), segment)


Tk().withdraw() # Simple GUI to select a file
filename = askopenfilename()
# print(os.path.splitext(os.path.basename(filename)))
if filename.endswith(".jpg"):
    split_image(filename, 480, 720)
elif filename.endswith(".mp4"):
    video_capture(filename, 300)
else:
    print("invalid extension")
