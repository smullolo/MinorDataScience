import cv2
import os
import csv
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


def add_to_array(to_array, from_array):
    for i in from_array:
        to_array.append(i)
    return to_array


def video_capture(videolink, frameinterval):
    vidcap = cv2.VideoCapture(videolink)  # load video
    success, image = vidcap.read()  # check if video is read succesfully
    count = 0  # variable to count the frames.
    framecap = frameinterval  # set interval for saving frames.
    name, extension = os.path.splitext(os.path.basename(videolink)) # split filename in file and extension.
    width = 720
    height = 510
    upper_border = 30

    name_array = []
    mean_array = []
    median_array = []
    mini_array = []
    maxi_array = []
    std_array = []

    while success:
        if count % framecap == 0:  # check if the framecount can be devided by 300
            cropped = image[upper_border:height, 0:width]
            cv2.imwrite("frames/%s_frame%d.jpg" % (name, count / framecap), cropped)  # save image every 300th frame
            # split_image("frames/%s_frame%d.jpg" % (name, count / framecap), 480, 720)

            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            cv2.imwrite("frames/gray_%s_frame%d.jpg" % (name, count / framecap), gray)  # save image every 300th frame
            temp_name_array, temp_mean_array, temp_median_array, temp_std_array, temp_mini_array, temp_maxi_array = split_image("frames/gray_%s_frame%d.jpg" % (name, count / framecap), 480, 720)

            name_array = add_to_array(name_array, temp_name_array)
            mean_array = add_to_array(mean_array, temp_mean_array)
            median_array = add_to_array(median_array, temp_median_array)
            std_array = add_to_array(std_array, temp_std_array)
            mini_array = add_to_array(mini_array, temp_mini_array)
            maxi_array = add_to_array(maxi_array, temp_maxi_array)

        success, image = vidcap.read()
        print("read a new frame succes:", success)
        count += 1

    # name_array = np.array(name_array)
    # mean_array = np.array(mean_array)
    # median_array = np.array(median_array)
    # std_array = np.array(std_array)
    # mini_array = np.array(mini_array)
    # maxi_array = np.array(maxi_array)
    # np.savetxt("excel_bestanden/%s.csv" % name, (mean_array, median_array, std_array, mini_array, maxi_array), header="Name, Mean, Median, Std, Mini, Maxi")
    # with open("excel_bestanden/%s.csv" % name, 'r') as f:


def split_image(image_source, height, width): # load image and split it into squares.
    image = cv2.imread(image_source)
    name, extension = os.path.splitext(os.path.basename(image_source))
    divider = gcd(height, width)
    y_length = int(height / divider)
    x_length = int(width / divider)

    name_array = []
    mean_array = []
    median_array = []
    mini_array = []
    maxi_array = []
    std_array = []

    for x in range(x_length):
        for y in range(y_length):
            y_start = y * divider
            x_start = x * divider
            segment = image[y_start:(y_start + divider), x_start:(x_start + divider)]
            cv2.imwrite("samples/%s_segment(%d,%d).jpg" % (name, x, y), segment)
            final_name = "%s_segment(%d,%d)" % (name, x, y)
            b, g, r = cv2.split(segment)
            b = np.array(b * 255)
            b = b.flatten()

            name_array.append(final_name)
            median = np.median(b)
            median_array.append(median)
            mean = np.mean(b)
            mean_array.append(mean)
            mini = np.min(b)
            mini_array.append(mini)
            maxi = np.max(b)
            maxi_array.append(maxi)
            std = np.std(b)
            std_array.append(std)

            # Print all the data:
            print("____________________________")
            print("name: %s" % final_name)
            print("mean: %d" % mean)
            print("median: %d" % median)
            print("smallest value: %d" % mini)
            print("Largest value: %d" % maxi)
            print("standard deviation: %d" % std)
            print("____________________________")

            with open("data/%s_segment(%d,%d)_arrays.txt" % (name, x, y), 'a') as f:
                np.savetxt(f, b, fmt='%i', delimiter='\t')
                f.close()
    return name_array, mean_array, median_array, std_array, mini_array, maxi_array


testarray1 = [1, 2, 3, 4, 5]
testarray2 = [1, 2, 4, 4, 5]
testarray2 = add_to_array(testarray1, testarray2)
arr = np.array(testarray2)
print(np.dstack((np.arange(1, arr.size+1), arr)))
print(np.dstack((np.arange(1, arr.size+1), arr))[0])
np.savetxt("foo.csv", np.dstack((np.arange(1, arr.size+1), arr))[0], "%d,%d", header="Id,Values")
# print(arr)

Tk().withdraw()  # Simple GUI to select a file
filename = askopenfilename()
# print(os.path.splitext(os.path.basename(filename)))
if filename.endswith(".jpg"):
    split_image(filename, 480, 720)
elif filename.endswith(".mp4"):
    video_capture(filename, 300)
else:
    print("invalid extension")

