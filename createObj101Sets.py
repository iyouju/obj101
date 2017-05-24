# -*- coding: utf-8 -*-
"""
Created on Mon May 22 08:58:42 2017

@author: iyouju
"""

import os
import random
import numpy as np
#from scipy import io #io.savemat

#import tensorflow as tf
from PIL import Image

#-------------------------------
IMG_H = 224     #图片高度
IMG_W = 224     #图片宽度
IMG_CH = 1      #图片通道数
LABEL_W = 102     #标签宽度

PROPORTION_TRAIN_DATA = 0.8     #训练集图片所占比例
path = '/home/iyouju/pythonPro/DATASETS/101_ObjectCategories/'  #数据集目录
#--------------------------------
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

#
def addImg2File(fileName,imgList,labelList):
    l = len(imgList)
    
    imgSet = []
    labSet = []
    ind = range(l)
    random.shuffle(ind)    
    
    imgNum = 0
    for i in ind:
        imgName = imgList[i]
        print(imgName)        
        
        index = labelList[i]
        imgRaw = Image.open(path+imgName)       
        temp = imgRaw.resize((IMG_W, IMG_H),Image.BILINEAR)
        img_arr = np.asarray(np.uint8(temp))
        sh = img_arr.shape
        if len(sh)==3:
            imgGray = rgb2gray(img_arr)
            imgSet.append(imgGray)
            labSet.append(index)
            imgNum += 1
        else:
            imgSet.append(img_arr)
            labSet.append(index)
            
    infor = {'IMG_NUM':imgNum,
            'IMG_H':IMG_H,
            'IMG_W':IMG_W,
            'IMG_CH':IMG_CH,
            'LABEL_W':LABEL_W}
    np.savez(fileName,img=imgSet,lab=labSet,info=infor)
#    np.savez(fileName,img=imgSet,lab=labSet)
    
    return imgNum
#    print('imgNum:%d' %imgNum)

#create tfrecords
def createObj101Sets():
    imgClassList = []   #各类图片名列表
    imgClassList = os.listdir(path)
    
    #遍历图片名列表，并将图片存入训练集和测试集中 
    trainLabelList = [] #训练集标签
    trainImgList = []  #训练图片列表  imgClass+'/'+imgName
    testLabelList = []  #测试集标签
    testImgList = []   #测试图片列表  imgClass+'/'+imgName
    
    iClass = 0    
    for imgClass in imgClassList:
        imgList = os.listdir(path+imgClass) #当前类中所有图片名列表
        listLen = len(imgList)      #当前类中图片数
        trainNum = np.floor(listLen * PROPORTION_TRAIN_DATA)  #当前类中训练集图片数，去尾舍入
        trainNum = np.int(trainNum)        
        
        
        #设置标签为独热码
        label = np.zeros(LABEL_W,dtype = int)
        label[iClass] = 1
        #将当前类中图片的标签及路径添加到列表中
        for i in range(trainNum):
            trainLabelList.append(label)
            
        temp = imgClass+'/'
        for imgName in imgList[0:trainNum]:
            trainImgList.append(temp+imgName)
        
        for i in range(listLen-trainNum,listLen):
            testLabelList.append(label)
        
        temp = imgClass+'/'
        for imgName in imgList[trainNum:]:
            testImgList.append(temp+imgName)
        
        iClass += 1
    
   
    trainNum = addImg2File("obj101Train",trainImgList,trainLabelList)
<<<<<<< HEAD
    testNum = addImg2File("obj101Test",testImgList,testLabelList)
=======
    testNum = addImg2File("objTest",testImgList,testLabelList)
>>>>>>> parent of 2b2a6bd... simAlexNet-can't converge (reverted from commit 5f9fe95e9da337f2547ea24d5771b6bec44a0fd2)
    
    
#
#    imgPath = []
#    imgLabel = []
#    for img in negNameList:
#        imgPath.append(negPath+img)
#        imgLabel.append(NEGLABEL)
#    for img in posNameList:
#        imgPath.append(posPath+img)
#        imgLabel.append(POSLABEL)
#    numTotal = len(imgPath)
#    numTrainData = np.floor(numTotal * PROPORTION_TRAIN_DATA)
##    numTestData = numTotal - numTrainData
#    
#    index = range(numTotal)
#    random.shuffle(index)
#    pathListTrain = []
#    labelListTrain = []
#    pathListTest = []
#    labelListTest = []
#    
#    cnt = 0
#    for i in index:
#        if cnt < numTrainData:
#            pathListTrain.append(imgPath[i])
#            labelListTrain.append(imgLabel[i])
#        else:
#            pathListTest.append(imgPath[i])
#            labelListTest.append(imgLabel[i])
#        cnt += 1
#     
#    trainNum = addImg2File("trainData_128x64",pathListTrain,labelListTrain)
#    testNum = addImg2File("testData_128x64",pathListTest,labelListTest)
#    print('trianNum:%d,testNum:%d' %(trainNum,testNum))
#    print('negLen:%d\tposLen:%d' %(len(negNameList),len(posNameList)))

#-------------------------------

if __name__ == '__main__':
    createObj101Sets()
    print('createObj101Sets is over.')
