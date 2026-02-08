import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
img = cv2.imread(r'D:\Python\Scripts\Pal_sir_work\roughentropy\Lena1.jpg',0)
#cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Define Lower_Object, Upper_Object, Lower_Background, Upper_Background
max_gray = np.max(img)
min_gray = np.min(img)
Gray_Levels=256
F=img
#Define the Bins
Lower_Object = np.zeros(Gray_Levels)
Upper_Object = np.zeros(Gray_Levels)
Lower_Background = np.zeros(Gray_Levels)
Upper_Background = np.zeros(Gray_Levels) 
Object_Roughness = np.zeros(Gray_Levels)
Background_Roughness = np.zeros(Gray_Levels)
Rough_entropy = np.zeros(Gray_Levels)
Part_1 =  np.zeros(Gray_Levels)
Part_2 =  np.zeros(Gray_Levels)
#Granule splitting (Divide_Image_in_to_Granules)
Size_of_Granule=int(input('enter the size of granule :'))
No_of_Granules=round(len(img)/Size_of_Granule)
#print(No_of_Granules)
#For Granule_1
for i in range(1,No_of_Granules):
    for k in range(1,No_of_Granules):
        Granule=img[Size_of_Granule*(i-1):Size_of_Granule*i, Size_of_Granule*(k-1):Size_of_Granule*k]
        #print(np.shape(Granule_1))
        Max_Granule=np.max(Granule)
        Min_Granule=np.min(Granule)
        #print(i,k)
#Compute Object_Low, Object_upper, BackGround_upper, Background_lower, 
        for j in range(0,255):
            if Max_Granule <= j <= max_gray:
                Lower_Object[j] = Lower_Object[j]+1
            if Min_Granule <=j<=max_gray-1:
                Upper_Object[j]=Upper_Object[j]+1
            if min_gray<=j<=Min_Granule:
                Lower_Background[j]=Lower_Background[j]+1
            if min_gray<=j<=Max_Granule:
                Upper_Background[j]= Upper_Background[j]+1
#Compute lower and upper approximations of Object and Background
for l in range (0,255):
    Object_Roughness[l] = 1 - (Lower_Object[l]/Upper_Object[l])
    Background_Roughness[l] = 1 - (Lower_Background[l]/Upper_Background[l])
    Part_1[l] = Object_Roughness[l]*np.log(Object_Roughness[l])
    Part_2[l] = Background_Roughness[l]*np.log(Background_Roughness[l])
    Rough_entropy[l]=float(-(2.718/2))*float((Part_1[l]+Part_2[l]))
#Maximum threshold
#index = Rough_entropy.index(max_value)
index = int(np.argmax(Rough_entropy))
# print(index)
Rough_entropy[0] = 0 
max_value = np.max(Rough_entropy)
#T=np.where(Rough_entropy == max_value)
T = int(np.nanargmax(Rough_entropy))
print(T)

F[img < T] = 0
F[img >= T] = 255

cv2.imshow("Segmented Image", F)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Simple thresholding
#for i in range (0,397):
#    for j in range(0,396):
#      if img[i,j]< T:
#         F[i,j] = 0 
#      else:
#         F[i,j] = 255
#cv2.imshow('Image',F)


