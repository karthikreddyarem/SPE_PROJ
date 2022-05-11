import numpy as np
import cv2
from drowsiness_detection import *

import unittest
import warnings 
import logging

logging.basicConfig(filename='drowsiness_detection.log',format='%(asctime)s %(message)s')
logger=logging.getLogger()
logger.setLevel(logging.INFO)

print("------------Testing on sample videos------------")

dict = {0:'No Drowsiness',1:'Drowsiness'}

class TestModel(unittest.TestCase):
    def test1(self):    
        rpred=[99]
        lpred=[99]
        i = 0
        thresh_time =0
        cum_score=[]    
        result = []
        cap = cv2.VideoCapture('test_videos/test1.mp4')
        while(True):
            success,frame = cap.read()
            if(success == 0):
                break
            frame_processed,i,cum_score,result = main_fun(frame,i,rpred,lpred,thresh_time,cum_score,result)

        if 1 in result:
            result = 1
        else:
            result = 0
        print("video 1 result:- " , " Predicted Status : " , dict[result], " , Actual Status: " , dict[1])
        logger.info("video 1 result:- %d , Predicted Status : %s , Actual Status : %s",result,dict[result],dict[1])
        
        self.assertEqual(result, 1)
        self.assertNotEqual(result, 0)
      
      
    def test2(self): 
        rpred=[99]
        lpred=[99]
        i = 0
        result = []
        thresh_time =0
        cum_score=[]    
        cap = cv2.VideoCapture('test_videos/test2.mp4')
        while(True):
            success,frame = cap.read()
            if(success == 0):
                break
            frame_processed,i,cum_score,result = main_fun(frame,i,rpred,lpred,thresh_time,cum_score,result)

        if 1 in result:
            result = 1
        else:
            result = 0
        
        print("video 1 result:- ", " Predicted Status : " , dict[result], " , Actual Status: " , dict[1])
        logger.info("video 1 result:- %d , Predicted Status : %s , Actual Status : %s",result,dict[result],dict[1])
        self.assertEqual(result, 1)
        self.assertNotEqual(result, 0)

    def test3(self): 
        rpred=[99]
        lpred=[99]
        i = 0
        thresh_time =0
        result = []
        cum_score=[]    
        cap = cv2.VideoCapture('test_videos/test3.mp4')
        while(True):
            success,frame = cap.read()
            if(success == 0):
                break
            frame_processed,i,cum_score,result = main_fun(frame,i,rpred,lpred,thresh_time,cum_score,result)

        if 1 in result:
            result = 1
        else:
            result = 0
        
        print("video 1 result:- ", " Predicted Status : " , dict[result], " , Actual Status: " , dict[0])
        logger.info("video 1 result:- %d , Predicted Status : %s , Actual Status : %s",result,dict[result],dict[0])
        self.assertEqual(result, 0)
        self.assertNotEqual(result, 1)

    def test4(self): 
        rpred=[99]
        lpred=[99]
        i = 0
        thresh_time =0
        cum_score=[]  
        result = []  
        cap = cv2.VideoCapture('test_videos/test4.mp4')
        while(True):
            success,frame = cap.read()
            if(success == 0):
                break
            frame_processed,i,cum_score,result = main_fun(frame,i,rpred,lpred,thresh_time,cum_score,result)

        if 1 in result:
            result = 1
        else:
            result = 0

        print("video 1 result:- ", " Predicted Status : " , dict[result], " , Actual Status: " , dict[0])
        logger.info("video 1 result:- %d , Predicted Status : %s , Actual Status : %s",result,dict[result],dict[0])
        self.assertEqual(result, 0)
        self.assertNotEqual(result, 1)
      

if __name__ == '__main__':
    unittest.main()
    
