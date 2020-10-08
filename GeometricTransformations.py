# -*- coding: utf-8 -*-
"""

 Pontificia Universidad Javeriana. Departamento de Electrónica
 Authors: Juan Henao, Marian Fuentes; Estudiantes de Ing. Electrónica.
 Procesamiento de Imagenes y video
 08/10/2020
"""

###################### GEOMETRIC TRANSFORMATIONS ######################################

#Click event function
#Detect the click button to save three pixel points
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(x,",",y)
        pixelPoints.append([x,y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x)+", "+str(y)
        #Show pixel coordinates on the image
        cv2.putText(image, strXY, (x,y), font, 0.3, (255,255,0), 1)
        cv2.imshow("image", image)    

#Calculate norm L1 of error between I1 similarity transform pixel points and I2 pixel points
def AccError(refImage, Image):
    rows = refImage.shape[0]  # check if both images have the same dimention
    cols = refImage.shape[1]
    Image = np.array(Image, dtype=np.float64) / 255 #Convert Images to float dt so error can be calculated
    refImage = np.array(Image, dtype=np.float64) / 255
    # pre_locate error array
    RGB_Errors = np.zeros(3, dtype=np.float64)
    if Image.shape[0] != rows or Image.shape[1] != cols:
        Image = cv2.resize(Image, (rows, cols), interpolation=cv2.INTER_CUBIC) # in other case resize second image
    # compare each pixel in every RGB component
    for i in range (rows):
        for j in range (cols):
            # Accumulate error
            RGB_Errors[0] = abs( RGB_Errors[0] + (refImage[i,j,0] - Image[i,j,0]) )
            RGB_Errors[1] = abs(RGB_Errors[1] + (refImage[i, j, 1] - Image[i, j, 1]))
            RGB_Errors[2] = abs(RGB_Errors[2] + (refImage[i, j, 2] - Image[i, j, 2]))

    TotalError = np.sum(RGB_Errors) #Sum all the errors in each RGB component
    return TotalError


# Main Code
if __name__ == "__main__":
    
    #Read path from image 1 (I1) and image 2 (I2)
    print("Path example: C:/Users/ACER/Desktop/Documents/lena.png")
    path1=input("Enter first path")
    path2=input("Enter second path")

    #List to save pixel points clicked on the image
    pixelPoints = []
    
    #Show image 1 to be clicked
    image = cv2.imread(path1)
    cv2.imshow("image", image)
    print("Click on the image three points. Then, close the image.")
    #calling the mouse click event
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Save the three pixel points clicked on the image 1 (I1)
    pixelpoints1 = pixelPoints
    pixelPoints = [] #clear pixelPoints 
    
    #Show image 2 to be clicked
    image = cv2.imread(path2)
    cv2.imshow("image", image)
    print("Click on the image another three points. Then, close the image.")
    #calling the mouse click event
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #Save the three pixel points clicked on image 2 (I2)
    pixelpoints2 = pixelPoints

    #print pixel points of image 1 and 2
    print("The six chosen points are: ")
    print(pixelpoints1, pixelpoints2)

    #Affine Transform
    pts1 = np.float32([pixelpoints1[0], pixelpoints1[1], pixelpoints1[2]])
    pts2 = np.float32([pixelpoints2[0], pixelpoints2[1], pixelpoints2[2]])
    #Save images, image 1 (I1), image 2 (I2)
    image = cv2.imread(path1)
    image2 = cv2.imread(path2)
    
    #Calculate Affine Transform matrix
    M_affine = cv2.getAffineTransform(pts1, pts2)
    #Apply Affine Transform to image 1 (I1)
    image_affine = cv2.warpAffine(image, M_affine, image.shape[:2])
    cv2.imshow("Affine Transform Image", image_affine)
    cv2.waitKey(0)

    #Similarity Transform (Affine Transform approximation)

    #Calculate parameters sx (s0), sy (s1), theta(rad), tx(x0) and ty(x1)
    sx = math.sqrt(M_affine[0,0]**2+M_affine[1,0]**2)
    sy = math.sqrt(M_affine[0,1]**2+M_affine[1,1]**2)
    theta = -np.arctan(M_affine[1,0]/M_affine[0,0])
    theta_rad = theta * np.pi / 180
    tx = (M_affine[0,2]*np.cos(theta)-M_affine[1,2]*np.sin(theta))/sx
    ty = (M_affine[0,2]*np.sin(theta)+M_affine[1,2]*np.cos(theta))/sy

    #Create Transalation (Tt), Sacale (Ts) and Rotation (Tr) Matrix 
    Tt = [[1,0,tx],[0,1,ty],[0,0,1]]
    Ts = [[sx,0,0],[0,sy,0],[0,0,1]]
    Tr = [[np.cos(theta_rad),np.sin(theta_rad),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]]

    #Dot multiplication Tr*Ts*Tt
    aux = np.dot(Tr,Ts)
    M_sim_calculated = np.dot(aux,Tt)

    #print(M_sim_calculated)
    #print(M_affine)
    
    #Calculate Similarity Transform matrix
    M_sim = np.float32([[sx * np.cos(theta_rad), -np.sin(theta_rad), tx],
                            [np.sin(theta_rad), sy * np.cos(theta_rad), ty]])
    
    #Apply similarity transform to image 1
    image_similarity = cv2.warpAffine(image, M_sim, image.shape[:2])
    cv2.imshow("Image Similarity", image_similarity)
    cv2.imshow("Image Affine", image_affine)
    cv2.imshow("Image Warped", image2)
    cv2.waitKey(0)

    #Calculate norm l1 of error
    # Both Images must be in RGB Color space
    Error = AccError(image2,image_similarity)
    print("Error is: ", Error)