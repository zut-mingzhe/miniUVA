# %%
#!/usr/bin/python
# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import serial
import numpy as np
import time
from threading import Thread
import threading
import csv
import DataDeal265 as dd
import struct
import queue
import torch
import cv2
import numpy as np
import ipywidgets
import pyzbar.pyzbar as pyzbar
import math
from cnn import *
from IPython.display import display



cnn=CNN()#加载网络
# 加载网络模型
cnn.load_state_dict(torch.load('weight/pytorchp.pkl',map_location='cpu'))
CopterTakingOff =1
TargetPosition = [0.0,0.0,0.0]
#PID参数P I D
pid=[0.1, 0.15, 0.1]
#图像大小
cap_x=360
cap_y=240
# cp1=[0.0,0.0]#起点
# cp2=[-1.0,0.0]#数字识别点一
# # cp3=[-1,1]#二维码识别点
# X坐标 Y坐标
#cp1=[0.0,0.0]#起点
# cp2=[-4.1,0.3]#数字识别点一
# cp3=[-2.0,2.2]#二维码识别点
# #左边路径
# cp4=[-2.5,1.7]
# cp5=[-2.5,3.5]
# cp6=[-2.0,4.0]
# #右边路径
# cp7=[-1.5,1.7]
# cp8=[-1.5,3.5]
# cp9=[-2.0,4.0]

# cp10=[-2.0,6.0]#数字识别点二
# cp11=[0.25,6.7]#降落点一
# cp12=[-5.0,6.0]#降落点二

cp0=[0.0,0.0]#起点
#左边路径
cp1=[-1.5,1.9]#数字识别点一
cp2=[-1.5,3.8]#数字识别点二
cp3=[-1.5,5.8]#数字识别点三

cp4=[-0.3,4.8]#二维码识别点
#右边路径
cp5=[1.5,5.8]#数字识别点四
cp6=[1.5,3.8]#数字识别点五
cp7=[1.5,1.9]#数字识别点六

cp8=[1.5,0.3]#右降落点
cp9=[-1.5,0.3]#左降落点
cp10=[0.0,1.0]


#智慧农业
# cp1=[0.0,0.0]#起点
# cp2=[0.0,3.9]#数字识别点一
# cp3=[-2.0,3.9]#二维码识别点
# #左边路径
# cp4=[-2.0,0.0]
# cp5=[0.0,0.0]
# cp6=[-2.0,0.0]
roqu=queue.Queue()
roqu.put(cp0)
roqu.put(cp1)
roqu.put(cp2)
roqu.put(cp3)
roqu.put(cp4)

font_scale=1.5 #字体大小
font=cv2.FONT_HERSHEY_PLAIN#字体类型
imageFlag=True
CE=0.3#坐标误差判断
KNOWN_DISTANCE = 10#飞行高度分米


# %%
class Pid():
    def __init__(self,  kp, ki, kd):
        self.KP = kp
        self.KI = ki
        self.KD = kd
        self.now_err = 0
        self.last_err = 0
        self.last_last_err = 0
        self.change_val = 0
        self.out_val=0
    def cmd_pid(self,exp_val,now_val):
        self.last_last_err = self.last_err
        self.last_err = self.now_err
        self.now_err = exp_val - now_val
        self.change_val = self.KP * (self.now_err - self.last_err) + self.KI * \
            self.now_err + self.KD * (self.now_err - 2 * self.last_err
                                      + self.last_last_err)
        self.out_val += self.change_val
        return self.out_val
# 字符串对比
def StrComparison(str1,str2):
    n = len(str1)
    res = []
    for x in str1:
        if x in str2:
            res.append(x)
    #print (n)
    return (n-len(res))
#二维码识别
def QRCode():
    global lst1
    signRight=0
    signLeft=0
    right =False
    left = False
    while True:
        ret,img_QR=cap.read()
        gray = cv2.cvtColor(img_QR, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)
        for barcode in barcodes:# 循环读取检测到的条形码
            # 绘条形码、二维码多边形轮廓
            points =[]
            for point in barcode.polygon:
                points.append([point[0], point[1]])
            points = np.array(points,dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(img_QR, [points], isClosed=True, color=(0,0,255),thickness=2)

            # 条形码数据为字节对象，所以如果我们想把它画出来
            # 需要先把它转换成字符串
            barcodeData = barcode.data.decode("UTF-8") #先解码成字符串
            lst1 = []
            lst2 = []
            #data_list = barcodeData.split(',')
            for i, num in enumerate(barcodeData.split(',')):
                if i<3:
                  lst1.append(num)
                else:
                  lst2.append(num)
            my_list = lst2[0]            
            # 绘出图像上的条形码数据和类型

            # print(barcodeData)
            cv2.putText(img_QR, barcodeData, (barcode.polygon[0].x, barcode.polygon[0].y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if StrComparison(str('right'),my_list )<=1:
                signRight+=1
                if signRight == 5:
                    right = True
                    print('right')


            if StrComparison( str('left'),my_list)<=1:
                signLeft+=1
                if signLeft == 5:
                    left = True
                    print('left')
        if left or right:
            break
        wid.value = cv2.imencode('.jpg',img_QR)[1].tobytes()
    return lst1,right,left

#寻找Roi区域
def findRoi(frame, thresValue,kernel):
    ret, ddst = cv2.threshold(frame,thresValue,255,cv2.THRESH_BINARY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.dilate(gray,kernel,iterations=2)
    gray2 = cv2.dilate(gray1,kernel,iterations=2)
    gray3 = cv2.erode(gray2,kernel,iterations=2)
    gray4 = cv2.dilate(gray3,kernel,iterations=2)
    # gray2 = cv2.erode(gray1,None,iterations=2)
    # ret, ddst = cv2.threshold(gray1,thresValue,255,cv2.THRESH_BINARY_INV)
    # ret, ddst = cv2.threshold(gray1,thresValue,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    # contours, hierarchy = cv2.findContours(ddst,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(gray4,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #contours储存检测到的轮廓
    area = []
    rois = []
    #len(contours)获取检测到的轮廓数量,它返回一个整数
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))   #计算每个轮廓的面积，并存入area列表
    if area  :
        max_idx = np.argmax(np.array(area))     #argmax()函数获取数组中最大元素的索引
        x, y, w, h = cv2.boundingRect(contours[max_idx])   #获取最大轮廓的左上角点的坐标、矩形的宽度和高度
        rois.append((x,y,w,h))
    return rois, ddst,area

#在roi区域中寻找数字
def findDigit(cnn, roi, thresValue):
    ret, th = cv2.threshold(roi, thresValue, 255, cv2.THRESH_BINARY)
    # ret, th = cv2.threshold(roi,thresValue,255,cv2.THRESH_OTSU)
    th = cv2.resize(th,(28,28),interpolation=cv2.INTER_AREA)     #将阈值化后的图像缩放为28x28大小的图像
    newing=[[th]]     #将缩放后的图像封装成一个二维列表，用于后续的神经网络预测
    newing = torch.Tensor(np.array(newing)).float()/255.
    predicions=cnn(newing)
    retult=torch.argmax(predicions).detach().numpy()
    return retult,th
#确定数字，储存数字
def mnist():
    global digits
    #Addo=0
    #Adde=0
    #odd=False
    #even=False
    #while True:
    digits = []
    if digit not in digits:
        ret,frame=cap.read()
        rois,edges,area = findRoi(frame, thresValue,kernel)
        
        if rois:
            x, y, w, h=rois[0]
            #裁剪roi，使区域减小，方便识别
            x_small=int(x+(w-h)/2)
            if  x_small<0:
                    x_small = 0
            # digit,th = findDigit(cnn,edges[y:y+h,x:x+w], 70)
            digit,th = findDigit(cnn,edges[y:y+h, x_small: x_small+h], thresValue)
            digit=int(digit)
            digits.append(digit)
    
            cv2.rectangle(frame, (int(x+(w-h)/2),y), (int(x+(w+h)/2),y+h), (153,153,0), 2)
            cv2.putText(frame, str(digit), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,0,255), 2)
            #cv2.putText(frame, str(digit), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,0,255), 2)
            newEdges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            newFrame = np.hstack((frame,newEdges))
            wid.value = cv2.imencode('.jpg',newFrame)[1].tobytes()
        return digits
    else:
        return
#协调修正,降落
def CoordinateCorrection(thresValue,kernel):
    global tarx
    global tary
    global routeNodeIndex
    
    while True:
        ret, frame=cap.read()
        rois, ddst = findRoi(frame,thresValue, kernel)
        if rois:
            x, y, w, h = rois[0]
            x_small = int(x + (w - h) / 2)
            if x_small < 0:
                x_small = 0
            width = (((x + w / 2) - (frame.shape[1] / 2)) * KNOWN_DISTANCE) / 520
            high = (((y + h / 2) - (frame.shape[2] / 2)) * KNOWN_DISTANCE) / 520
            tarx = tarx + width
            cv2.rectangle(frame, (x, y), (int(x + w), y + h), (153, 153, 0), 2)
            cv2.putText(frame, "%.2fft" % width, (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
            sum_x += width / 10
            sum_y += high / 10
            conter += 1
        newEdges = cv2.cvtColor(ddst, cv2.COLOR_GRAY2BGR)
        newFrame = np.hstack((frame, newEdges))
        wid.value = cv2.imencode('.jpg',newFrame)[1].tobytes()
        if conter == 10:
            tarx=tarx+sum_x/10
            tary=tary+sum_y/10
        routeNodeIndex+=1
        print("检测到的降落点:x = %.1f, y = %.1f" % (routeNodeIndex,tarx, tary))
            # 在此处添加代码，根据检测到的目标坐标调整无人机的降落点
            # 例如，可以使用比例控制根据误差信号调整无人机的位置
        break   
    return tarx,tary
def massdrop():
    global tarx
    global tary
    global routeNodeIndex
    center_points = []
    while len(center_points) < 10:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        rect = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center_points.append((int(x + w/2), int(y + h/2)))

    center_points = np.array(center_points)
    avg_center_point = tuple(np.mean(center_points, axis=0, dtype=np.int))
    tarx=avg_center_point[0]
    tary=avg_center_point[1]
    routeNodeIndex+=1
print ("route node %d: x : %.1f , y : %.1f  " %(routeNodeIndex,tarx,tary))
#路径点更新图像识别线程
def Router():
    global timer
    global routeNodeIndex
    global SendTargetPos
    global CopterLanding
    global LaserArray
    global LaserDistance
    global FlightMode
    global CopterTakingOff
    global routeStartFlag
    global tarx
    global tary
    global routerFlag
    global dataBuf
    global abc
    

    while True:
        SendTargetPos = 1
        
        if math.sqrt((tary-pos_Y)**2+(tarx-pos_X)**2)<CE and routeStartFlag and routerFlag:
            if imageFlag:
            #数字识别启动判断
                if math.sqrt((cp1[1]-pos_Y)**2+(cp1[0]-pos_X)**2)<CE:
                    #数字识别
                    digits = mnist()
                if math.sqrt((cp2[1]-pos_Y)**2+(cp2[0]-pos_X)**2)<CE:
                    digits = mnist()    
                if math.sqrt((cp3[1]-pos_Y)**2+(cp3[0]-pos_X)**2)<CE:
                    digits = mnist()   
                if math.sqrt((cp5[1]-pos_Y)**2+(cp5[0]-pos_X)**2)<CE:
                    digits = mnist()   
                if math.sqrt((cp6[1]-pos_Y)**2+(cp6[0]-pos_X)**2)<CE:
                    digits = mnist()   
                if math.sqrt((cp7[1]-pos_Y)**2+(cp7[0]-pos_X)**2)<CE:
                    digits = mnist()   

                #二维码识别启动判断
                if math.sqrt((cp4[1]-pos_Y)**2+(cp4[0]-pos_X)**2)<CE:
                    #二维码识别
                    lst1,right,left=QRCode()
                    while lst1!=None:
                        roqu.put(cp5)
                        if digits[3]==lst1[0]:
                            
                            lst1=lst1[1:]
                        roqu.put(cp6)
                        if digits[4]==lst1[0]:
                            lst1=lst1[1:]
                        roqu.put(cp7)
                        if digits[5]==lst1[0]:
                            lst1=lst1[1:]              
                        roqu.put(cp10)
                        roqu.put(cp1)
                        if digits[0]==lst1[0]:
                            lst1=lst1[1:]
                        roqu.put(cp2)
                        if digits[1]==lst1[0]:
                            lst1=lst1[1:]
                        roqu.put(cp3)
                        if digits[2]==lst1[0]:
                            lst1=lst1[1:]                                                                                
                        
                    if lst1==None:
                        if left:
                            #左边路径入队
                            if pos_X<0:
                                roqu.put(cp6)
                                roqu.put(cp10)
                                roqu.put(cp9)
                                
                                print('left')
                            if pos_X>0:
                                roqu.put(cp9)                      
                                print('left')
                        if right:
                            if pos_X>0:
                                roqu.put(cp1)
                                roqu.put(cp10)
                                roqu.put(cp8)
                                
                                print('right')
                            if pos_X<0:
                                roqu.put(cp8)                      
                                print('right') 
                        
            if math.sqrt((cp1[0]-pos_Y)**2+(cp1[1]+pos_X)**2)<CE:
                abc=0x01
                while abc==0x01:
                    routeNodeIndex+=1
                    tarx=float(cp1[0])
                    tary=float(cp1[1])
                    print ("route node %d: x : %.1f , y : %.1f  " %(routeNodeIndex,tarx,tary))
                    if Continue:
                        Continue=False
                        break
            if not roqu.empty():
                if abc==0x00 and Continue==False:
                    TargetPosition=roqu.get()
                    routeNodeIndex+=1
                    tarx=float(TargetPosition[0])
                    tary=float(TargetPosition[1])
                    print ("route node %d: x : %.1f , y : %.1f  " %(routeNodeIndex,tarx,tary))
                    time.sleep(1)
                 
            else:
                              # pipe.stop()
                #2SendTargetPos = 0
                tarx,tary=CoordinateCorrection(frame,thresValue,kernel)
                CopterTakingOff = 1
                if math.sqrt((pos_X-tarx)**2+(pos_Y-tary)**2)<10:
                    CopterLanding = 1
                routeNodeIndex= 1
                closeRouter = True
                print ("Landing")
                routerFlag=False
#物块投放
def massdrop():
    global massdropping
    while massdropping==True:
        ret,frame = cap.read()
        kernel = np.ones((4, 4), np.uint8)
        gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.medianBlur(gray,3)
        ret,dst=cv2.threshold(gray2,125,255,cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, 20,param1=100, param2=30, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        #circles[0]圆心的 x 坐标,circles[1]圆心的y坐标，circles[2]圆的半径大小
        
    lst1=lst1[1:]

#串口通信线程
def PortCom(port):
    global pipe  
    global cfg
    global SendTargetPos
    global CopterLanding
    global CopterTakingOff
    global _265Ready
    global GetOnceCmd
    global routeNodeIndex
    global routeStartFlag
    global closeRouter
    global abc
    global Continue
    closeRouter = False
    Continue = False
    while True:
#         size=port.inWaiting()
#         if(size!=0):
        if closeRouter:
            router.cancel() #于取消一个名为"router"的任务
        response = port.readline() #从端口读取一行数据，并将其赋值给变量"response"
        if response !=None:
            port.flushInput()  #清空端口的输入缓存
            CmdStr1 = str(b'Start265\n')  
            CmdStr2 = str(b'Departures\n')
            CmdStr3 = str(b'Refresh265\n')
            CmdStr4 = str(b'put_in_ok\n')
            CmdStr5 = str(b'high_back_up_ok\n')
            CMD = str(response)
            #刷新265
            if( ( StrComparison( CMD ,CmdStr1 )<=1)  and  GetOnceCmd ==False):
                print(StrComparison( CMD ,CmdStr1 ),response,CMD)
                # 声明RealSense管道，封装实际的设备和传感器（初始化）
                pipe = rs.pipeline()
#                 try:
#                     pipe.stop()
#                 except:
#                     print("Error1")
                #构建配置对象并请求姿态数据
                cfg = rs.config()
                cfg.enable_stream(rs.stream.pose,rs.format.any,framerate=200)
                # 开始流与请求的配置
                pipe.start(cfg)
                dd.initData()  #初始化姿态解算器的数据缓存
                SendTargetPos = 0
                CopterLanding = 0
                _265Ready=True
                GetOnceCmd =True
                routeStartFlag = True

            elif( ( StrComparison(CMD ,CmdStr2 )<=1)  and  CopterTakingOff == 1 ):
                print(StrComparison(CMD ,CmdStr2 ),response,CMD)
                print("Get!")
                router = Thread(target=Router, args=())
                router.start()
                CopterTakingOff = 0
            elif( StrComparison(CMD ,CmdStr4 )<=1):
                print(StrComparison(CMD ,CmdStr4 ),response,CMD)
                print("OK!")
                abc = 0x00
            elif( (StrComparison(CMD ,CmdStr5 )<=1) and abc==0x00):
                print(StrComparison(CMD ,CmdStr5 ),response,CMD)
                print("Continue!")
                Continue = True
                
                
            elif( StrComparison(CMD ,CmdStr3 )<=1):
                _265Ready = False
                GetOnceCmd = False
                routeNodeIndex= 1
                CopterTakingOff =1
                routeStartFlag = False
                print("ReStart!")
                print( StrComparison(CMD ,CmdStr3 ),response,CMD)
                try:
                    pipe.stop()
                    time.sleep(1.0)
                except:
                    print("Error2")
            response = 0
            CMD = 0
        time.sleep(0.02)


if __name__ == '__main__':
    global routeNodeIndex   #路径节点索引
    global SendTargetPos    #发送目标位置
    global CopterLanding    #着陆
    global LaserArray
    global LaserDistance
    global FlightMode
    global pipe
    global _265Ready
    global GetOnceCmd
    global CheckSum
    global tarx
    global tary
    global routerFlag
    global dataBuf
    global abc
    port = serial.Serial(port="/dev/ttyAMA0",baudrate=230400,stopbits=1,parity=serial.PARITY_NONE,timeout=1000)
    kernel = np.ones((4, 4), np.uint8) #膨胀算子
    thresValue=50                      #阈值
    wid = ipywidgets.Image() #创建一个Jupyter Notebook中的图像小部件（widget），以便在笔记本中显示图像
    #打开摄像头
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        cap=cv2.VideoCapture(1)
    if not cap.isOpened():
        raise IOError('Can not open video')
    #设置摄像头参数
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,cap_x)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cap_y)#将视频帧高度设置为cap_y
    ret,frame = cap.read()

    #串口通信线程
    thread_Serial = Thread(target=PortCom, args=(port,))
    thread_Serial.start()

    #导入，目标位置坐标
    TargetPosition=roqu.get()
    tarx=TargetPosition[0]
    tary=TargetPosition[1]

    routeNodeIndex = 1
    _265Ready =False
    GetOnceCmd =False
    CheckSum =0
    dataBuf = [0]*65
    #激光阵列？
    LaserArray =0
    #激光测距？
    LaserDistance=0.0
    #飞行模式
    FlightMode =0
    routerFlag =True
    abc=0x00
    display(wid)
    try:                #try:用于处理特定类型的异常
        while True:
            # ret,Frame = cap.read()
            if _265Ready:
                # Wait for the next set of frames from the camera
                frames = pipe.wait_for_frames()
                # Fetch pose frame
                pose = frames.get_pose_frame()
                if pose:
                    # Print some of the pose data to the terminal
                    data = pose.get_pose_data()
                    #pos_X、pos_Y、pos_Z分别是无人机在X、Y、Z三个方向上的位置信息
                    #Euler是无人机当前的欧拉角信息，包括俯仰角、横滚角和偏航角
                    dataBuf,pos_X,pos_Y,pos_Z,Euler=dd.solveData(data)
                    if SendTargetPos == 1:
                        # posZ = TargetPosition[2]
                        posZ=0.0
                        dataBuf[43] = 0x20
                        posX_buf = struct.pack("f",tarx)
                        dataBuf[44] = posX_buf[0]
                        dataBuf[45] = posX_buf[1]
                        dataBuf[46] = posX_buf[2]
                        dataBuf[47] = posX_buf[3]
                        posY_buf = struct.pack("f",tary)
                        dataBuf[48] = posY_buf[0]
                        dataBuf[49] = posY_buf[1]
                        dataBuf[50] = posY_buf[2]
                        dataBuf[51] = posY_buf[3]
                        posZ_buf = struct.pack("f",posZ)
                        dataBuf[52] = posZ_buf[0]
                        dataBuf[53] = posZ_buf[1]   
                        dataBuf[54] = posZ_buf[2]
                        dataBuf[55] = posZ_buf[3]

                        dataBuf[56] = LaserArray
                        Laser_Dis = struct.pack("f",LaserDistance)
                        dataBuf[57] = Laser_Dis[0]
                        dataBuf[58] = Laser_Dis[1]
                        dataBuf[59] = Laser_Dis[2]
                        dataBuf[60] = Laser_Dis[3]
                        dataBuf[61] = FlightMode

                    if CopterLanding==1:
                        dataBuf[62] = 0xA5
                    else:
                        dataBuf[62] = 0x00

                    for i in range(0,62):
                        CheckSum =  CheckSum + dataBuf[i]

                    dataBuf[63] = 0xAA
                    dataBuf[64] = CheckSum & 0x00ff

                    print("\rrpy_rad[0]:{:.2f},rpy_rad[1]:{:.2f},rpy_rad[2]:{:.2f} ,X:{:.2f},Y:{:.2f},Z:{:.2f} ".format(Euler[0]*57.3,Euler[1]*57.3,Euler[2]*57.3,pos_Y,-pos_X,pos_Z),end="")
                    port.write(dataBuf)
                    #校验和
                    CheckSum = 0
#                     pipe.stop()
            else:
                dataBuf[0] = 0x55
                dataBuf[1] = 0xAA
                dataBuf[2] = 0xFF
                dataBuf[63] = 0xAA
                dataBuf[64] = 0x00
                port.write(dataBuf)
                time.sleep(0.1)
            # wid.value = cv2.imencode('.jpg',Frame)[1].tobytes()
    finally:           #finally:块中的代码始终会在try:块中的代码执行结束后执行，无论是否发生异常。

        pipe.stop()
        dataBuf[62] = 0xA5
        port.write(dataBuf)
        print("some erro")
#         pipe.stop()


