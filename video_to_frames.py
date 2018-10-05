import cv2

vidcap = cv2.VideoCapture('DJI_0001.mp4')  # load video
success, image = vidcap.read()  # check if video is read succesfully
count = 0  # variable to count the frames.
framecap = 300  # set interval for saving frames.

width = 720
height = 513
upper_border = 28

while success:
    if count % framecap == 0:  # check if the framecount can be devided by 300
        cropped = image[upper_border:height, 0:width]
        # gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("frame%d.jpg" % (count / framecap), cropped)  # save every 300 th frame

    success, image = vidcap.read()
    print("read a new frame suces:", success)
    count += 1
