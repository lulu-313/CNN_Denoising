import numpy as np
from PIL import Image

#拆分图片
def sliding_window(im,vx,vy,dx,dy):
   
    #偏移量
    dx = dx
    dy = dy
    n = 1

    #左上角切割
    x1 = 0
    y1 = 0
    x2 = vx
    y2 = vy

    #纵向
    patches = []
    while x2 <=im.size[1]:
        #横向切
        while y2 <= im.size[0]:
#            name3 = name2 + str(n) + ".bmp"
            #print (n,x1,y1,x2,y2)
            im2 = im.crop((y1, x1, y2, x2))
            im2 = np.reshape(im2, (vx, vy))
#            im2.save(name3)
            y1 = y1 + dy
            y2 = y1 + vy
            patches.append(im2)
            n = n + 1
            
        x1 = x1 + dx
        x2 = x1 + vx
        y1 = 0
        y2 = vy

#    print ("图片切割成功，切割得到的子图片数为",n-1)
    patch=np.array(patches)
    return patch

def sliding_window1(im,vx,vy):
   
    #偏移量
    dx = 60
    dy = 60
    n = 1

    #左上角切割
    x1 = 0
    y1 = 0
    x2 = vx
    y2 = vy

    #纵向
    patches = []
    while x2 <=im.size[1]:
        #横向切
        while y2 <= im.size[0]:
#            name3 = name2 + str(n) + ".bmp"
            #print (n,x1,y1,x2,y2)
            im2 = im.crop((y1, x1, y2, x2))
            im2 = np.reshape(im2, (vx, vy))
#            im2.save(name3)
            y1 = y1 + dy
            y2 = y1 + vy
            patches.append(im2)
            n = n + 1
            
        x1 = x1 + dx
        x2 = x1 + vx
        y1 = 0
        y2 = vy

    print ("图片切割成功，切割得到的子图片数为",n-1)
    patch=np.array(patches)
    return patch
#拼接图片
def CombineImage(picture,height,weight,row,col,dim):
    toImage=Image.new("RGB",(weight,height))
    t=0
    for i in range(col):
        for j in range(row):
            loc=(j*dim,i*dim)
            img = Image.fromarray(picture[t])
            toImage.paste(img, loc)
            t += 1
    return toImage


#保存训练评分
import xlrd; 
from xlutils.copy import copy
def SaveScore(test_name,loss,accuracy):
 oldWb = xlrd.open_workbook('D:/projects/pytnon/lyc/model/eval_result.xls')
 table = oldWb.sheets()[0]
 nrows = table.nrows #行数
 nrows

 newWb = copy(oldWb); 
 #print newWb; # 
 newWs = newWb.get_sheet(0); 
 newWs.write(nrows, 0, test_name ); 
 newWs.write(nrows, 1, loss ); 
 newWs.write(nrows, 2, accuracy ); 
 #print "write new values ok"; 
 newWb.save('D:/projects/pytnon/lyc/model/eval_result.xls'); 
 print("评估结果保存成功！！！")
 
 
 #加载数据
from skimage import io,color
def convert_gray(f):
    rgb=io.imread(f)
    return color.rgb2gray(rgb)
def load_imgs(str):     
    x_train = io.ImageCollection(str,load_func=convert_gray)
    height=x_train[0].shape[0]
    width=x_train[0].shape[1]
    
    x_train=io.concatenate_images(x_train)
    x_train= x_train.astype('float32') / 255.
    x_train = np.reshape(x_train, (-1, height, width, 1))
    return x_train

#信噪比
def psnr(im1,im2):    
    im1,im2=np.matrix(im1).astype('float64'),np.matrix(im2).astype('float64')
    diff = np.abs(im1 - im2)
    mse = ((np.square(diff)).sum())/(diff.shape[0]*diff.shape[1])
    psnr = 20*np.log10(255/np.sqrt(mse))
    return psnr

def Savepsnr(test_name,psnr):
 oldWb = xlrd.open_workbook('D:/projects/pytnon/lyc/model/psnr.xls')
 table = oldWb.sheets()[0]
 nrows = table.nrows #行数
 nrows

 newWb = copy(oldWb); 
 #print newWb; # 
 newWs = newWb.get_sheet(0); 
 newWs.write(nrows, 0, test_name ); 
 newWs.write(nrows, 1, psnr ); 
 #print "write new values ok"; 
 newWb.save('D:/projects/pytnon/lyc/model/psnr.xls'); 
 print("信噪比保存成功！！！")
 
 #扩充边框  
import cv2
#以最外边界向外扩充
def Copy_Border(img,width):
    img_n=[]
    for i in range(img.shape[0]):
        img_n.append(cv2.copyMakeBorder(img[i],width,width,width,width,cv2.BORDER_REPLICATE))
    img_n=np.array(img_n)
    return img_n
#图片外边界为轴反转扩充
def Copy_Border1(img,width):
    img_n=[]
    for i in range(img.shape[0]):
        img_n.append(cv2.copyMakeBorder(img[i],width,width,width,width,cv2.BORDER_REFLECT))
    img_n=np.array(img_n)
    return img_n

def Cut_Border(img,width):
    decoded_imgs_n=[]
    for i in range(img.shape[0]):
        decoded_imgs_n.append(img[i][width:(img.shape[1]-width),width:(img.shape[1]-width)])
    decoded_imgs=np.array(decoded_imgs_n)
    return decoded_imgs


import struct
def read_filedata(filename_path):
    data_img=[]
    f=open(filename_path,'rb')  
    #f.seek(0,0)
    f.seek(0, 0)
    temp=b"11"
    while True:
        string=''
        temp=f.read(1)
        if temp==b'':
            break
        s1=str(hex(ord(temp)))[2:]
        if len(s1)==1:
            s1="0"+s1
        temp=f.read(1)
        if temp==b'':
            break
        s2=str(hex(ord(temp)))[2:]
        if len(s2)==1:
            s2="0"+s2
        temp=f.read(1)
        if temp==b'':
            break
        s3=str(hex(ord(temp)))[2:]
        if len(s3)==1:
            s3="0"+s3
        temp=f.read(1)
        if temp==b'':
            break
        s4=str(hex(ord(temp)))[2:]
        if len(s4)==1:
            s4="0"+s4
        string=s4+s3+s2+s1
        data_img.append(struct.unpack('!f', bytes.fromhex(string))[0])
    f.close()
    return data_img

#5——12 数据换位矩阵操作
def matrix_split(data_img,vx,vy,dx,dy):  
    #vx分块后数据长，vy分块后数据宽，dx方向西东的长度，dy宽方向移动的长度
    #偏移量
    dx = dx
    dy = dy
    n = 1

    #左上角切割
    x1 = 0
    y1 = 0
    x2 = vx
    y2 = vy

    #纵向
    patches = []
    while x2 <= data_img.shape[1]:
        #横向切
        while y2 <= data_img.shape[0]:
#            name3 = name2 + str(n) + ".bmp"
#            print (n,x1,y1,x2,y2)
            im2 = data_img[y1:y2,x1:x2]
            im2 = np.reshape(im2, (vx, vy))
#            im2.save(name3)
            y1 = y1 + dy
            y2 = y1 + vy
            patches.append(im2)
            n = n + 1
            
        x1 = x1 + dx
        x2 = x1 + vx
        y1 = 0
        y2 = vy

#    print ("数据切割成功，切割得到的子数据数为",n-1)
    patch=np.array(patches)
    return patch
def matrix_combin(data_img,x,y):  
   #x 行数 ；y 列数
   #偏移量
   a=np.zeros((y,x))
   
   vx = data_img.shape[1]
   vy = data_img.shape[2]
        
   dx = vx
   dy = vy
   n = 1
        
   #左上角切割
   x1 = 0
   y1 = 0
   
        
   x2 = vx
   y2 = vy
   #纵向
       
   while x2 <= a.shape[1]:#3000
       #横向切
       while y2 <= a.shape[0]:#300
#           print (n,x1,x2,y1,y2)  
           a[y1:y2,x1:x2] = data_img[n-1] 
           y1 = y1 + dy
           y2 = y1 + vy
           n = n + 1

       x1 = x1 + dx
       x2 = x1 + vx
       y1 = 0   
       y2 = vy
   return a
def normal(data_img):
    min_data=min(data_img)
    max_data=max(data_img)
    data_com=[]
    for i in range(len(data_img)):
        #归一化#离差标准化转化
        data_com.append(((data_img[i]-min_data)/(max_data-min_data))) 
    return data_com
