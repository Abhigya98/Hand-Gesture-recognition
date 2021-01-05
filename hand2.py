import cv2
import numpy as np
import math
#import device as dev

cap = cv2.VideoCapture(0)

while(1):

    try:

        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)

        roi = frame[100:300, 100:300]
        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
        #cv2.imshow('Frame',frame)
        #cv2.VideoWriter('Frame.mp4',frame,20.0,(640,480))
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        
        lower_skin = np.array([0,20,70], dtype = np.uint8)
        upper_skin = np.array([20,255,255], dtype = np.uint8)
        # lower_skin = np.array([0,48,80], dtype = np.uint8)
        # upper_skin = np.array([20,255,255], dtype = np.uint8)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        # cv2.imshow('f0',mask)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        
        mask = cv2.dilate(mask,kernel,iterations=2)
        cv2.imshow('f2',mask)
        # mask = cv2.erode(mask,kernel,iterations=2)
        # cv2.imshow('f1',mask)
        # mask = cv2.dilate(mask,kernel,iterations=1)
        # cv2.imshow('f3',mask)
        mask = cv2.GaussianBlur(mask,(3,3),0)#50)
        cv2.imshow('frame',mask)


        #mask = cv2.dilate(mask,kernel,iterations=3)
        #cv2.imshow('dilate',mask)

        #mask=cv2.erode(mask,kernel,iterations=2)
        #cv2.imshow('erode',mask)
        
        cnts = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnt = max(cnts, key = lambda x: cv2.contourArea(x))
        
        cv2.drawContours(roi,[cnt], -1,(0,255,0),1)
    
        hull = cv2.convexHull(cnt)
        #cv2.drawContours(roi,[hull],-1,(0,0,255),1,8)
        #cv2.imshow("12",roi)
        areahull = cv2.contourArea(hull)
        # print(areahull)   
        areacnt = cv2.contourArea(cnt)
        # print(" area cnt = ",areacnt)
        arearatio=((areahull-areacnt)/areacnt)*100
        # print("area ratio = ",arearatio)
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        #print(epsilon)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        hull=cv2.convexHull(approx,returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        #print(defects[0])
        #cv2.drawContours(roi,defects,-1,(255,0,0),1,8)
        cv2.imshow('13',roi)

        l=0

        for i in range(defects.shape[0]):
            s,e,f,d =defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))

            d=(2*ar)/a
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
            
            if angle <= 90 and d>30:
                l += 1
                cv2.circle(roi, far, 3, [255,0,0], -1)
            # print(l)
            #draw lines around hand
            #cv2.line(roi,start, end, [0,255,0], 2)
            cv2.imshow('roi2',roi)
            
        l+=1
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l==1:
            
            if areacnt<2000:
                cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                if arearatio<17.5:
                    cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                else: #arearatio<17.5:
                    print("hand gesture = ", 1)
                    cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                '''   
                else:
                    print("hand gesture = ",BL)
                    cv2.putText(frame,'Best of luck',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

                '''
        elif l==2:
            cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
            
        elif l==3:
         
              if arearatio<80:
                    cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
              #else:
                    #cv2.putText(frame,'ok',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    
        elif l==4:
            cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==5:
            cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            #dev.post_image()
            #print("Helllooo")
        elif l==6:
            cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        else :
            cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        
        cv2.imshow('fram2',frame)
    except:
        pass
            
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break

cv2.destroyAllWindows()
cap.release()
