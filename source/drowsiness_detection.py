import cv2
from keras.models import load_model
import keras
import numpy as np
import logging
import warnings

face = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar_cascade_files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar_cascade_files/haarcascade_righteye_2splits.xml')

model = load_model('models/cnncat2.h5')
score=0

output_annotations = []
cum_score = []



logging.basicConfig(filename='drowsy_detections.log',format='%(asctime)s %(message)s')
logger_info=logging.getLogger()
logger_info.setLevel(logging.INFO)

def main_fun(img,i,rpred,lpred,thresh_time,cum_score,output):
        label_val=1
        height,width,layers = img.shape 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.3,minSize=(40,40))
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)


        size = (width,height)
         
        
        for (x,y,w,h) in right_eye:
            r_eye=img[y:y+h,x:x+w]
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=  r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = np.argmax(model.predict(r_eye), axis=-1)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
            break

        for (x,y,w,h) in left_eye:
            l_eye=img[y:y+h,x:x+w]
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = np.argmax(model.predict(l_eye), axis=-1)
            cv2.rectangle(img, (x,y) , (x+w,y+h) , (0,0,255) , 1)
            break

        if(rpred[0] == 0 and lpred[0] == 0):
            score = 1
            label_val = 0
        else:
            score = 0   

        cum_score.append(score)
        if(len(cum_score) > 11):#11
            thresh_time = np.sum(cum_score[i-9:i])

        label_txt = {0:"closed eyes",1:"open eyes"}
        label_colors = {0:(0,0,255),1:(0,255,0)}
        i = i+1
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y) , (x+w,y+h) ,label_colors[label_val], 3)
            cv2.rectangle(img,(x,y-30),(x+w,y),label_colors[label_val],-1)
            cv2.putText(img,label_txt[label_val],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
        cv2.putText(img,'cum%:'+str((thresh_time*100/10)),(50,height-20),cv2.FONT_HERSHEY_SIMPLEX,1,(165,255,0),1)
        if(thresh_time>=6):
            cv2.putText(img,'DROWSINESS',(300,height-20),cv2.FONT_HERSHEY_SIMPLEX,1,(165,255,0),2)
            logger_info.info("Status:- Drowsiness State with Threshold time %d", thresh_time)
            output.append(1)
        else:
            logger_info.info("Status:- Drowsiness not detected with threshold time %d", thresh_time)
            output.append(0)
        return img,i,cum_score,output

