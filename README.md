# DNN_Denoising <br>

　　地震勘探数据采集过程中多方面影响，使得到的地震数据都是含有有效信号和噪声的混波记录。 <br>
噪声的存在使得有效的地震数据很难被识别，因此在实际应用中去除噪声干扰是地震数据处理流程　 <br>
中的重要步骤。为解决在去噪的过程中需要人为设定去噪阈值，实现起来较为复杂，效率较低等问  <br>
题，提供一种深度学习模型训练方法和地震数据去噪方法以实现高效、智能化地进行地震数据去噪 <br>
的效果。

* 去噪模型
　　地震数据去噪模型以卷积自动编码器为原型 <br>
　　将输入样本压缩到隐层，然后解压，在输出端重建样本 <br>
　　通过网络学习将冗余信息去掉，将有用信息输入到隐层中，通过隐层的特征信息重建得到无噪声的图片 <br> 
