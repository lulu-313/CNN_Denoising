# 代码实现 <br>
#### 其中util为工具包，
* 包括16进制数地震数据转换为图片 <br>
* 滑动切片函数
* 图片边缘扩展（边缘翻转对称填充）
* 正确率损失值计算
* 图结构保存
* 信噪比计算等

### 16进制数地震数据转换为图片
　　对地震数据进行处理，实现地震数据可视化。我们为大家准备的数据的存储形式是十六进制数据，<br>
但其实际行形式是四个字节的浮点数，所以我们第一步要将十六进制数据转化为四字节浮点数，我们  <br>
还要在此基础上将浮点数数据标准化和灰度化，然后将其转化为更方便我们理解的图片形式。下图就 <br>
是地震数据的原始形态。 <br>

但是要注意的是假定原始数据顺序为1234 5678,则正确的读入顺序应为：7856 3412，这是由于我们 <br>
使用python读取地震图片数据，如果我们利用C语言读取数据则按照1234 5678的顺序读取就可以了。<br>
![image](https://github.com/lulu-313/DNN_Denoising/blob/master/image/%E5%8E%9F%E6%95%B0%E6%8D%AE.png)<br> 

#### 其中显示隐层为保存隐层特征提取过程，
