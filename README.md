# img2img
图像翻译，顾名思义，就是想语言翻译一样，从一种图像转换为另一种图像，例如把二维地图转换为三维地图，把模糊图像转换为清晰图像，把素描转化为彩图，下面是图像翻译的几个例子：
![img1](https://github.com/lsqpku/img2img/blob/master/doc/blur2clear.png)
![img2](https://github.com/lsqpku/img2img/blob/master/doc/capes.png)
![img３](https://github.com/lsqpku/img2img/blob/master/doc/facades.png)

在这个项目中，我们使用一个简单的神经网络实现模糊图像的清晰化处理，展示一下处理效果和有趣发现，最后介绍一个GAN在图像翻译的最新应用。
note: 本项目中的一部分代码来自于DCGAN项目(https://github.com/Newmu/dcgan_code)。

参考自编码神经网络,把GAN中discriminator和generator组合起来，形成一个图像翻译网络，网络结构如下：
![network] (https://github.com/lsqpku/img2img/blob/master/doc/network.jpg)

我们把DCGAN中的discriminator改造为encoder,把DCGAN中的generator改造为decoder, 这一结构类似于自编码器，不同的是，在训练这个翻译网络的时候，输入和输出图像术语不同的域。例如在训练一个模糊图像清晰化的翻译网络时，输入训练样本是模糊图像，输出训练样本要尽量和对应的清晰图像接近，因此训练样本必须是每一张模糊样本有一张对应的清晰样本。前面第一副图像就是使用该神经网络对测试样本进行的清晰化处理，使用的是wiki face的彩色男性人脸，使用高斯模糊处理得到模糊图像，这样就得到一大批对偶的模糊－清晰图像训练集（超过3万张）。从清晰化处理效果看，虽然不完美，但清晰化程度大大提高。

那么使用男性人脸训练的模型在对女性人脸进行清晰化处理的效果如何呢？下图是女性测试样本的处理效果（同样对女性样本进行高斯模糊），和男性处理效果基本一致。

![image_female] (https://github.com/lsqpku/img2img/blob/master/doc/wiki_female.png)

开一下脑洞，对非人脸样本的处理效果如何？下图是对一些非人脸图片的测试效果（同样是先进行了高斯模糊），发现效果也是比较好的，但是色系发生了变化，从冷色调转变为人脸色调。
![image_building] (https://github.com/lsqpku/img2img/blob/master/doc/wiki_building.png)
把上图转为灰度图像，清晰化效果看起来更明显一些：
![image_building] (https://github.com/lsqpku/img2img/blob/master/doc/wiki_building_gray.png)

然而，高斯模糊只是模糊方法的一种，使用高斯模糊图像进行训练的模型是否适用于其他模糊方案（例如现resize原图像的1/16,再resize回原尺寸）？下图是对resize模糊方案处理过的测试样本的测试效果：
![image_building] (https://github.com/lsqpku/img2img/blob/master/doc/wiki_resize_blur.png)
