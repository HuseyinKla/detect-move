import cv2 as cv
import numpy as np

bg = cv.createBackgroundSubtractorMOG2()

minarea=1000

vid = cv.VideoCapture(0)

while 1:
    _,frame = vid.read()
    bgmask = bg.apply(frame,None,0.02)
    erode = cv.erode(bgmask,None,iterations=4)
    moments = cv.moments(erode,True)



    if moments['m00']>=minarea:
        erode=cv.putText(erode,"NESNE TESPIT EDILDI",(100,100),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)

    cv.imshow("dilate",erode)
    cv.imshow("frame",frame)


    if cv.waitKey(1) & 0xFF == ord("q"):
        break


vid.release()
cv.destroyAllWindows()