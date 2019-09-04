# =============================================================================
# 5月3日
# 采用滑动窗口的方式进行图片切片，图片切片保留交叉部分--解决图片拼接色差问题
#                               图片切割尺寸向外扩充--解决图片训练后存在边缘问题
#                               采用对整张图片先增加外边界--解决图片切割四个边界的切割尺寸问题
# =============================================================================


#模型改进版本
from keras import Input
import numpy as np
from keras.layers import MaxPooling2D, UpSampling2D, Conv2D
from keras.models import Model#,load_model
from PIL import Image
from utils import sliding_window,CombineImage,SaveScore,psnr,Copy_Border,Cut_Border
from skimage import io,color
from sklearn.model_selection import train_test_split  #用来分割训练集和测试集


if __name__=="__main__":
    
    #------加载数据--------
    #边界宽度
    border_width=30
    #切片宽度
    img_width=120
    #图片转灰度
    def convert_gray(f):
        rgb=io.imread(f)
        return color.rgb2gray(rgb)
    #加载噪声图并切片
    str_real='D:/python/p_image/noisy'+ '/*.bmp'
    x_noisy = io.ImageCollection(str_real,load_func=convert_gray)#
    x_noisy = io.concatenate_images(x_noisy)
    #扩充边界
    x_noisy = Copy_Border(x_noisy,border_width)
    #切片 大小为120
    for i in range(x_noisy.shape[0]):
        if i ==0:
            img1= Image.fromarray(x_noisy[i])
            noisy = sliding_window(img1, img_width, img_width,60,60)
        else:
            img1= Image.fromarray(x_noisy[i])
            noisy = np.concatenate((noisy,sliding_window(img1, img_width, img_width,60,60)))
    #加载原图并切片
    str_real='D:/python/p_image/train'+ '/*.bmp'
    x_train = io.ImageCollection(str_real,load_func=convert_gray)#
    x_train = io.concatenate_images(x_train)
    #扩充边界
    x_train = Copy_Border(x_train,border_width)
    #切片 大小为120
    for i in range(x_train.shape[0]):
        if i ==0:
            img1= Image.fromarray(x_train[i])
            real = sliding_window(img1, img_width, img_width,60,60)
        else:
            img1= Image.fromarray(x_train[i])
            real = np.concatenate((real,sliding_window(img1, img_width, img_width,60,60)))
            
    # 设定参数
    #批处理胡个数
    batch_size = 128
    # 训练轮数
    epoch = 50 
    # 图片的维度
    img_rows, img_cols = real[0].shape[1],real[0].shape[1] 
    # 卷积滤镜的个数
    nb_filters = 32
    # 最大池化，池化核大小
    pool_size = (2, 2)
    # 卷积核大小
    kernel_size = (3, 3)    
    
    # 切分数据
    X_train, X_test, Y_train, Y_test = train_test_split(noisy, real, test_size=0.2, random_state=42)
   
    # 使用TensorFlow的顺序：(conv_dim1,conv_dim2,conv_dim3,channels)
    X_train = np.reshape(X_train,(-1,img_rows, img_cols, 1))
    X_test = np.reshape(X_test,(-1, img_rows, img_cols, 1))
    Y_train = np.reshape(Y_train,(-1, img_rows, img_cols, 1))
    Y_test = np.reshape(Y_test,(-1, img_rows, img_cols, 1))
    #灰度值归一化
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    Y_train = Y_train.astype('float32') / 255.
    Y_test = Y_test.astype('float32') / 255.

    # 定义输入层
    input_img = Input(shape=(img_rows, img_cols, 1)) 

    # 定义encoder卷积层
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  
    x = MaxPooling2D((2, 2), padding='same')(x)  
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) 
    encoded = MaxPooling2D((2, 2), padding='same')(x) 
#    encoded = Dropout(0.25)(x)

    # 定义decoder卷积层
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded) 
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) 
    x = UpSampling2D((2, 2))(x)  # (?, 60, 60, 32)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    # 选定模型的输入，decoded（即输出）的格式
    denoise = Model(input_img, decoded)
    # 定义优化目标和损失函数
    denoise.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['accuracy'])

    #打印模型
    denoise.summary()
  
    
    flag=input("训练请输入a,预测结果请输入b,退出请输入c:")
#训练模型
    if flag=='a':
        flag=input("是否加载现有的训练结果(Y/N):")
        if flag=='y':
            name = "denoise_weight150.h5"
            name=input("请输入模型名称:")
            name+='.h5'
            denoise.load_weights("D:/projects/pytnon/lyc/model/"+name)
        flag = input("是否更改训练次数(Y/N):")
        if flag=='y':
            epoch=int(input("请输入训练次数:"))
        # 训练
        denoise.fit(X_train, Y_train,  # 输入输出
                epochs=epoch,  # 迭代次数
                batch_size=batch_size,#批处理
                shuffle=True,
                validation_data=(X_test, Y_test))  # 验证集
        #计算损失值和准确率
        score = denoise.evaluate(X_test, Y_test, verbose=0)
        flag=input("是否保存模型(Y/N):")
        if flag=='y':
            name1='temp_weight'
            name='temp.h5'
            flag=input("是否对模型进行命名(Y/N):")
            if flag=='y':
                name=input("请输入文件名:")
                name1=name+"_weight.h5"
                name+=".h5"
                
            #保存权重
            denoise.save_weights("D:/projects/pytnon/lyc/model/"+name1)
            
            #保存模型
            denoise.save("D:/projects/pytnon/lyc/model/"+name)
     
          
        print("保存成功!!!")
        print("显示评估结果")
        #loss和accuracy
        loss,accuracy=score[0],score[1]
        SaveScore(name,loss,accuracy)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        
        print("显示预测结果")
       
        # 加载预测图片
        flag = input("是否更改预测图片(Y/N):")
        imgname = "shot15.bmp"
        if flag == 'y':
            imgname = input("请输入图片名称:")
            imgname+='.bmp'
        path = "D:/python/p_image/noisy/" + imgname
        img = Image.open(path).convert("L")
        
        img = np.reshape(img,(-1,1800,1200))
        img1= Image.fromarray(np.reshape(Copy_Border(img,border_width),(1860,1260)))
        pre_img = sliding_window(img1, img_width, img_width,60,60)        
        
        pre = np.reshape(pre_img, (-1,img_cols,img_rows, 1))
        pre = pre.astype('float32') / 255.
        #进行预测
        decoded_imgs = denoise.predict(pre)
        decoded_imgs = np.reshape(decoded_imgs * 255., (-1, img_cols,img_rows))
        
       
        #去掉边界
        decoded_imgs=Cut_Border(decoded_imgs,border_width)
       
        #拼接图片
        raw_imgs = CombineImage(decoded_imgs, 1800, 1200, 20, 30, 60)
        raw_imgs.show()
        flag=input("是否保存图片(Y/N):")
        if flag=='y':
            name=input("请输入保存的图像名称:")
            name+=".bmp"
            raw_imgs.save('D:/python/p_image/img/'+name)
            
#调用模型进行预测           
    elif flag=='b':
        #直接预测结果
        name1='temp_weight'
        name="temp.h5"
        flag=input("是否输入要加载的模型(Y/N):")
        if flag=='y':
            name=input("请输入文件名:")
            name1=name+"_weight.h5"
            name+=".h5"
                
            #加载权重
            denoise.load_weights("D:/python/model/"+name1)
           
        # 加载预测图片
        flag=input("是否更改预测图片(Y/N):")
        imgname="shot15.bmp"
        if flag=='y':
            imgname=input("请输入图片名称:")
            imgname += ".bmp"
        path = "D:/python/p_image/noisy/"+imgname
        img = Image.open(path).convert("L")
        
        img = np.reshape(img,(-1,1800,1200))
        img1= Image.fromarray(np.reshape(Copy_Border(img,border_width),(1860,1260)))
        pre_img = sliding_window(img1, img_width, img_width,60,60) 
        
        pre = np.reshape(pre_img, (-1, img_cols,img_rows, 1))
        pre = pre.astype('float32') / 255.
        print("显示预测结果")
        #进行预测
        decoded_imgs = denoise.predict(pre)
        decoded_imgs = np.reshape(decoded_imgs * 255., (-1, img_cols,img_rows))
        
        #去掉边界
        decoded_imgs=Cut_Border(decoded_imgs,border_width)
        
        #拼接图片
        raw_imgs = CombineImage(decoded_imgs, 1800, 1200, 20, 30, 60)
        raw_imgs.show()
        flag = input("是否保存图片(Y/N):")
        if flag == 'y':
            name = input("请输入保存的图像名称:")
            name += ".bmp"
            raw_imgs.save('D:/python/p_image/img/' + name)
    else:
        #退出
        exit()
        
#计算信噪比        
path = 'D:/python/p_image/train/real.bmp'
real_imgs = Image.open(path).convert("L")  
#path = 'D:/projects/pytnon/lyc/p_image/noisy/shot10.bmp'
#noisy_imgs = Image.open(path).convert("L") 
raw_imgs=raw_imgs.convert("L")
psnr = psnr(real_imgs,raw_imgs)
print ("信噪比：",psnr)

