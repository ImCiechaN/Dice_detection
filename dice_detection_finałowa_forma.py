#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from os import listdir

import math
import matplotlib
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
from IPython.display import Video
from IPython.display import HTML

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


matplotlib.rcParams['figure.figsize'] = (20.0, 18.0)


# In[60]:


def capture_frames():
    """funkcja do przechwycenia klatek z filmów
    
    imput: lista nazw filmów
    output: lista przechwyconcyh klatek
    """
    cap = cv2.VideoCapture('test6big.mp4') #przechwytywanie filmu
    frameRate = cap.get(cv2.CAP_PROP_FPS) #zwraca ilość klatek na sek
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) #zwraca ilość klatek w filmie

    img_list = []
    while cap.isOpened():
        frameId = cap.get(cv2.CAP_PROP_POS_FRAMES) #zwraca obecny numer klakti
        ret, frame = cap.read() #jeżeli klatka jest wczytana poprawnie, zwraca True
        if (ret != True):
            break
        if (frameId % math.floor(frameRate/2.0) == 0):
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #konwertowanie klatki z BGR na GRAY
            img_list.append(gray_image)
    cap.release() #zamyka plik wideo
    return img_list

def detect_pips_and_locations(captured_frames):
    """ funkcja do obliczenia ilości oczek na najbardziej widocznej ściance oraz wykrycia pozycji kostek
    
    input: lista klatek
    output: każda klatka z naniesioną ilością oczek
    """
    
    for f in captured_frames:
        gray_image = f
        x_range1 = int(gray_image.shape[0]*0.06)#gray_image[0] -> zwraca rozmiar wierszy w macierzy
        x_range2 = int(gray_image.shape[0]*0.91)
        y_range1 = int(gray_image.shape[1]*0.05)#gray_image[1] -> zwraca rozmiar kolumn w macierzy
        y_range2 = int(gray_image.shape[1]*0.95)

        # usuwanie zewnętrznych granic klatki
        gray_image[:,0:y_range1] = 0.0
        gray_image[:,y_range2:] = 0.0
        gray_image[:x_range1,:] = 0.0
        gray_image[x_range2:,:] = 0.0

        plt.figure(figsize=(20,18))

        params = cv2.SimpleBlobDetector_Params() #klasa pozwalająca na ustawienie parametrów do wyjrycia "bloba"  
        params.filterByArea = True
        params.filterByCircularity = True
        params.filterByInertia = True
        params.minThreshold = 50
        params.maxThreshold = 300
        params.minArea = 25
        params.maxArea = 70
        params.minCircularity = .3
        params.minInertiaRatio = .4

        detector = cv2.SimpleBlobDetector_create(params) #stworzenie obiektu - detektora "bloba"
        keypoints = detector.detect(gray_image) #lista zawierająca wykryte "bloby"
        inv_image = cv2.bitwise_not(gray_image) #odwrócenie bitów w obrazie
        keypoints2 = detector.detect(inv_image) #druga lista zawierająca wykryte "bloby"
        im_with_keypoints = cv2.drawKeypoints(gray_image, keypoints+keypoints2, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #obrysowanie "bloba"

        plt.imshow(cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2RGB))

        thresh = 39
        X = np.array([list(i.pt) for i in keypoints+keypoints2])
        # grupowanie oczek tak by każda grupa należała do swojej kostki
        if len(X) > 0:
            clusters = hcluster.fclusterdata(X, thresh, criterion="distance")
            cluster_no = [np.sum(clusters==i) for i in clusters]
            num_dict = {np.where(clusters == i)[0][0]:np.sum(clusters==i) for i in np.unique(clusters)}
            key_map = {i:{np.sum(clusters==i):[X[np.where(np.array(clusters) == i)[0]]]} for i in np.unique(clusters)}
            for i,v in key_map.items():
                for j, k in v.items():
                    plt.text(k[0][0][0]+35, k[0][0][1]+18, s=str(j), fontsize=25, color='red')
                    
            plt.scatter(*np.transpose(X), c=clusters)
            plt.axis('off')
            #title = "threshold: %f, ilość grup: %d" % (thresh, len(set(clusters)))
        plt.show()


# In[61]:


captured_frames = capture_frames()
detect_pips_and_locations(captured_frames)


# In[ ]:





# In[ ]:




