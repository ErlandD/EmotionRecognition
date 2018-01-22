import dlib
import numpy as np
import os
from skimage import io

predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

'''img = io.imread("Train1/D2N2Sur/frame0.jpg")

dets = detector(img)
nums = 442

for k, d in enumerate(dets):
    shape = predictor(img, d)

vec = np.empty([68, 2], dtype = int)
for b in range(68):
    vec[b][0] = shape.part(b).x
    vec[b][1] = shape.part(b).y

with open('Train1/D2N2Sur.txt', 'a') as file:
    for a in vec:
        for b in a:
            file.write(str(b) + " ")
        file.write("\n")
'''
# trainpack = [1, 3, 4, 6, 7, 9, 11, 14, 20, 21, 23, 27, 28, 29, 30, 32, 34, 37, 38]
trainpack = [14, 20, 21, 23, 27, 28, 29, 30, 32, 34, 37, 38]
emotion = ['D2N2Sur', 'H2N2A', 'H2N2C', 'H2N2D', 'H2N2S', 'N2A', 'N2C', 'N2D', 'N2H', 'N2S', 'N2Sur', 'S2N2H']

for t in trainpack:
    for e in emotion:
        list = os.listdir('Train'+str(t)+'/'+e)
        frames = len(list)
        videovec = []
        for f in range(0, frames):
            # print(f)
            img = io.imread('Train'+str(t)+'/'+e+'/frame'+str(f)+'.jpg')
            dets = detector(img)
            for k, d in enumerate(dets):
                shape = predictor(img, d)

            vec = np.empty([68, 2], dtype=int)
            for b in range(68):
                vec[b][0] = shape.part(b).x
                vec[b][1] = shape.part(b).y
            videovec.append(vec)
        with open('Train'+str(t)+'/'+e+'.txt', 'w') as file:
            for vid in videovec:
                for coordinate in vid:
                    file.write(str(coordinate)+' ')
                file.write('\n')
