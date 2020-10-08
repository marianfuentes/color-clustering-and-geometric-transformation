# Pontificia Universidad Javeriana. Departamento de Electrónica
# Authors: Juan Henao, Marian Fuentes; Estudiantes de Ing. Electrónica.
# Procesamiento de Imagenes y video
# 08/10/2020

# Import Librarys.
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from math import sqrt
from numpy.core._multiarray_umath import ndarray
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

# Method for re-constructing clustered images, provided by Julian Quiroga
def recreate_image(centers, labels, rows, cols):
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d))
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image_clusters

# Method for calculating distances between every point (in RGB space) and it´s closest cluster center (for an image)
def CalcDistance(centers, labels, rows, cols, image, n):
    distSum = np.zeros(n, dtype=np.float64) # aux accumulative variable
    label_idx = int(0) # index aux variable
    for i in range(rows): #For every pixel in a given image
        for j in range(cols):
            aux = labels[label_idx] # closest cluster to the pixel
            x1 = centers[aux,0] # take every x,y,z (RGB component) of the center of the cluster
            y1 = centers[aux,1]
            z1 = centers[aux,2]
            x2 = image[i,j,0] #take every x,y,z (again in RGB space) of the given pixel.
            y2 = image[i,j,1]
            z2 = image[i,j,2]
            # Calculate distance between pixel and cluster center
            dist = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            distSum[aux] = distSum[aux] + dist # Accumulate (sum) every distance for each of n clusters
            label_idx = label_idx + 1 # next pixel

    TotalSum = np.sum(distSum) # sum all of the n cluster distances
    return TotalSum

# Main Code
if __name__ == "__main__":
    #########################################################################
    ####### 1. Receive Image path,name and choose segmentation method #######
    #########################################################################
    print("Path example: C:/Users/ACER/Desktop/Semestre10/Imagenes/Presentaciones/Semana 9/Imagenes")
    path = input("Please enter a path, there is an example above. ")
    print("name Example: bandera.png")
    name = input("Please enter an image name, there is an example above. ")
    path_name = os.path.join(path, name)
    image = cv2.imread(path_name) # read image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # change image to RGB color space
    method = input("Please enter kmeans or gmm to choose between color segmentation methods. ")

    #########################################################################
    ############### 2. Transform Image to a Numpy 2D Array ##################
    #########################################################################
    # to use the mat plot lib we need to change the image to a float np array and normalize it.
    image = np.array(image, dtype=np.float64) / 255
    rows, cols, ch = image.shape
    # "pre locate" a new image to re create a clustered image
    image_array = np.reshape(image, (rows * cols, ch))

    #########################################################################
    ############ 3. Fit model, get labels and recreate images ###############
    #########################################################################
    # take a sample of the image pixels, in this case 0.5% of the pixels image
    image_array_sample = shuffle(image_array, random_state=0)[:int(rows*cols*0.05)]
    # in this variable we will store the sum of distances for eache n = 1,2, ... 10 clusters
    distance = np.zeros(10)
    n = int(0) # n clusters
    # this code is not efficient so we give a head up to user, calculating distances takes a lot of time !
    print("Please wait while distances and images are computed, this takes some time, code has not stopped working")
    for f in range (10): # for n clusters
        n = n + 1 # next cluster
        if method == 'gmm':
            model = GMM(n_components=n).fit(image_array_sample) # train model
        elif method == 'kmeans':
            model = KMeans(n_clusters=n, random_state=0).fit(image_array_sample) # train model
        else:
            raise SyntaxError("Method must be gmm or kmeans, please try again")

        if method == 'gmm':
            labels = model.predict(image_array) # get model labels
            centers = model.means_ # get model center(s)
            distance[f] = CalcDistance(centers, labels, rows, cols, image, n) # calculate distances
        else:
            labels = model.predict(image_array) # get model labels
            centers = model.cluster_centers_ # get model center(s)
            distance[f] = CalcDistance(centers, labels, rows, cols, image, n) # calculate distances

        # here we just graph the clustered images, if you don´t understand something check up the mat plot lib doc
        plt.figure(n)
        plt.clf()
        plt.axis('off')
        plt.title('Quantized image ({} color cluster(s), method={})'.format((n), method))
        plt.imshow(recreate_image(centers, labels, rows, cols))

    # Graph the original image
    print("Showing Quitized Images and Original Image, Close them so the code can continue")
    plt.figure(11)
    plt.clf()
    plt.axis('off')
    plt.title('Original image')
    plt.imshow(image)
    plt.show()

    ####################################################################################
    ########### 4. Histogram of the distances for every n color cluster ################
    ####################################################################################
    # Finally we graph every n cluster for the total sum of the distance of each pixel and the nearest cluster center
    print("Showing sum of distances from every point to it´s nearest cluster")
    names = ['n = 1', 'n = 2', 'n = 3', 'n = 4', 'n = 5', 'n = 6', 'n = 7', 'n = 8', 'n = 9', 'n = 10']
    plt.figure(12)
    plt.clf()
    plt.title('Sum of distances to nearest Cluster, method={}'.format(method))
    plt.bar(names, distance)
    plt.ylabel("Sum of total distances for each Cluster")
    plt.xlabel("n number of color clusters")
    plt.show()

##############################################################################################################
# Pontificia Universidad Javeriana, Sede Bogota. #############################################################
# Authors: Juan Henao & Marian Fuentes. - Proc. de imagenes y video ##########################################
# ############################################################################################################
