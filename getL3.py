import numpy as np
import cv2

def getL3(imgd,camK,sd_len,Z):
    bord = 5
    imgd = imgd.astype('float')

    KK=camK;
    px = KK[0,0];
    py = KK[1,1];
    v0=KK[0,2]/2;
    u0=KK[1,2]/2;

    Gx_arr = np.zeros(imgd.shape)
    Gy_arr = np.zeros(imgd.shape)

    m = 0;
    Lsd = np.zeros((sd_len,6))
    xy_arr = np.zeros((sd_len,4))

    for i in range(bord, imgd.shape[0]-bord):
        for j in range(bord, imgd.shape[1]-bord):
            # Gx = (2047.0 *(imgd(i,j+1) - imgd(i,j-1))+913.0 *(imgd(i,j+2) - imgd(i,j-2))+112.0 *(imgd(i,j+3) - imgd(i,j-3)))/8418.0;
            Gx = (2047.0 *(imgd[i,j+1] - imgd[i,j-1])+913.0 *(imgd[i,j+2] - imgd[i,j-2])+112.0 *(imgd[i,j+3] - imgd[i,j-3]))/8418.0
            # Gy = (2047.0 *(imgd(i+1,j) - imgd(i-1,j))+913.0 *(imgd(i+2,j) - imgd(i-2,j))+112.0 *(imgd(i+3,j) - imgd(i-3,j)))/8418.0;
            Gy = (2047.0 *(imgd[i+1,j] - imgd[i-1,j])+913.0 *(imgd[i+2,j] - imgd[i-2,j])+112.0 *(imgd[i+3,j] - imgd[i-3,j]))/8418.0
            # Gx_arr(i,j)=Gx;
            Gx_arr[i,j]=Gx;
            # Gy_arr(i,j)=Gy;
            Gy_arr[i,j]=Gy;


            Ix=px*Gx
            Iy=py*Gy

            # Not sure check once

            # y = ((i - 1 - u0)/px)
            y = ((i - u0)/px)
            # x = double((j - 1 - v0)/py) ;
            x = ((j - v0)/py)

            Zinv =  1/Z


            xy_arr[m,0]=Ix
            xy_arr[m,1]=Iy
            xy_arr[m,2]=x
            xy_arr[m,3]=y

            Lsd[m,0] = Ix * Zinv
            Lsd[m,1] = Iy * Zinv
            Lsd[m,2] = -(x*Ix+y*Iy)*Zinv
            Lsd[m,3] = -Ix*x*y-(1+y*y)*Iy
            Lsd[m,4] = (1+x*x)*Ix + Iy*x*y
            Lsd[m,5]  = Iy*x-Ix*y
            m = m +1
    return m , Lsd
