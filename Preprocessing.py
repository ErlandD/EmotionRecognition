import cv2
from PIL import Image
import os

filename = 'H2N2A'
trainingfiles = 38
numframe = 1
#Zilli2012

#python 27, get frames
def getFrames():
    vidcap = cv2.VideoCapture('Train'+str(trainingfiles)+'/'+filename+'.MP4')
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        #print 'Read a new frame: ', success
        print filename,"getFrames:", count
        #print "Train"+str(trainingfiles)+"/"+filename+"/frame%d.jpg" % count, image
        cv2.imwrite("Train"+str(trainingfiles)+"/"+filename+"/frame"+str(count)+".jpg", image)
        count += 1
        #break
    vidcap.release()
    return count-1

# crop image
def cropImage():
    frame = 0
    while frame < numframe:
        img = Image.open("Train"+str(trainingfiles)+"/"+filename+"/frame"+str(frame)+".jpg")
        #area = (515, 275, 825, 640) #Train1
        #area = (470, 310, 780, 675) #Train3 14 29
        #area = (490, 255, 800, 620) #Train4 11
        #area = (490, 210, 800, 575) #Train 27 28 37
        #area = (500, 175, 810, 540) #Train 30
        #area = (465, 345, 775, 710) #Train7
        #area = (500, 255, 810, 620) #Train6 9 20
        #area = (490, 280, 800, 645) #Train 23
        #area = (490, 310, 800, 675) # Train 32 34
        area = (510, 310, 820, 675) #Train 38
        cropped_img = img.crop(area)
        cropped_img.save("Train"+str(trainingfiles)+"/"+filename+'cutted/frame'+str(frame)+'.jpg')
        print filename,"cropImage:", frame
        frame += 1
        #break

#motion detection
def motionDetection():
    frame = 1
    image = cv2.imread("Train"+str(trainingfiles)+"/"+filename+"cutted/frame0.jpg")
    newline = ""
    with open('Train'+str(trainingfiles)+'/'+filename+'.txt', 'w') as file:
        file.write("")
    while frame < numframe:
        imagen = cv2.imread("Train"+str(trainingfiles)+"/"+filename+"cutted/frame" + str(frame) + ".jpg")
        width = 0
        height = 0
        print filename,"motionDetection:", frame
        while height < 365:
            while width < 310:
                if image[height, width, 0] < imagen[height, width, 0] - 5 or image[height, width, 0] > imagen[
                    height, width, 0] + 5:
                    newline += str(1)
                elif image[height, width, 1] < imagen[height, width, 1] - 5 or image[height, width, 1] > imagen[
                    height, width, 1] + 5:
                    newline += str(1)
                elif image[height, width, 2] < imagen[height, width, 2] - 5 or image[height, width, 2] > imagen[
                    height, width, 2] + 5:
                    newline += str(1)
                else:
                    newline += str(0)
                width += 1
            width = 0
            height += 1
            with open('Train'+str(trainingfiles)+'/'+filename+'.txt', 'a') as file:
                file.write(newline + "\n")
            newline = ""
        frame += 1
        image = imagen

lfn = ['D2N2Sur', 'H2N2A', 'H2N2C', 'H2N2D', 'H2N2S', 'N2A', 'N2C', 'N2D', 'N2H', 'N2S', 'N2Sur', 'S2N2H']
#lfn = ['H2N2A', 'H2N2D', 'H2N2S', 'N2A', 'N2C', 'N2D', 'N2S', 'N2Sur', 'S2N2H']
for fn in lfn:
    filename = fn
    numframe = getFrames()
    cropImage()
    motionDetection()

#numframe = getFrames()
#getFrames()
#cropImage()
#motionDetection()

'''batch = [14, 20, 23, 27, 28, 29, 30, 32, 34, 37, 38]
for tf in batch:
    for fn in lfn:
        os.makedirs("Train"+str(tf)+"/"+fn)
        os.makedirs("Train" + str(tf) + "/" + fn + "cutted")
'''