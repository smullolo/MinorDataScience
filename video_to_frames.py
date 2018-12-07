import cv2
import os
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

    
def calculate_array_values(image_array):
    current_array = image_array
    smallest_value = 255
    largest_value = 0
    for i in current_array:
        if i < smallest_value:
            smallest_value = i
        if i > largest_value:
            largest_value = i
    mean = sum(current_array) / len(current_array)
    return mean, smallest_value, largest_value


def gcd(x, y):  # Calculate greatest common devider.
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
            cv2.imwrite("frames/%s_frame%d.jpg" % (name, count / framecap), cropped)  # save image every 300th frame
            # split_image("frames/%s_frame%d.jpg" % (name, count / framecap), 480, 720)

            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            cv2.imwrite("frames/gray_%s_frame%d.jpg" % (name, count / framecap), gray)  # save image every 300th frame
            split_image("frames/gray_%s_frame%d.jpg" % (name, count / framecap), 480, 720)
        success, image = vidcap.read()
        print("read a new frame succes:", success)
        count += 1


def split_image(image_source, height, width): # load image and split it into squares.
    image = cv2.imread(image_source)
    name, extension = os.path.splitext(os.path.basename(image_source))
    divider = gcd(height, width)
    y_length = int(height / divider)
    x_length = int(width / divider)

    for x in range(x_length):
        for y in range(y_length):
            y_start = y * divider
            x_start = x * divider
            segment = image[y_start:(y_start + divider), x_start:(x_start + divider)]
            cv2.imwrite("samples/%s_segment(%d,%d).jpg" % (name, x, y), segment)
            b, g, r = cv2.split(segment)
            b = np.array(b * 255)
            b = b.flatten()

            with open("data/%s_segment(%d,%d)_arrays.txt" % (name, x, y), 'a') as f:
                np.savetxt(f, b, fmt='%i', delimiter='\t')
                f.close()


Tk().withdraw() # Simple GUI to select a file
filename = askopenfilename()
# print(os.path.splitext(os.path.basename(filename)))
if filename.endswith(".jpg"):
    split_image(filename, 480, 720)
elif filename.endswith(".mp4"):
    video_capture(filename, 300)
else:
    print("invalid extension")

