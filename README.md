# img2img
图像翻译，顾名思义，就是想语言翻译一样，从一种图像转换为另一种图像，例如把二维地图转换为三维地图，把模糊图像转换为清晰图像，把素描转化为彩图，下面是几个例子：
!(https://github.com/lsqpku/img2img/tree/master/doc/blur2clear.png)
!(https://github.com/lsqpku/img2img/tree/master/doc/capes.png)
在这个项目中，我们使用一个简单的神经网络实现模糊图像的清晰化处理，展示一下处理效果和有趣发现，最后介绍一个GAN在图像翻译的最新应用。
note: 本项目中的一些代码从DCGAN项目中复制(https://github.com/Newmu/dcgan_code)。

参考自编码神经网络,把GAN中discriminator和generator组合起来，形成一个图像翻译网络，网络结构如下：




