# img2img
图像翻译，顾名思义，就是像语言翻译一样，把一种图像转换为另一种图像，例如把二维地图转换为三维地图，把模糊图像转换为清晰图像，把素描转化为彩图，下面是图像翻译的几个例子：
![img1](https://github.com/lsqpku/img2img/blob/master/doc/blur2clear.png)
![img2](https://github.com/lsqpku/img2img/blob/master/doc/capes.png)
![img3](https://github.com/lsqpku/img2img/blob/master/doc/facades.png)

在这个项目中，我们使用一个简单的神经网络实现模糊图像的清晰化处理，展示一下处理效果和有趣发现，最后介绍一个生成对抗神经网络（GAN）在图像翻译的应用。
note: 本项目中的一部分代码来自于DCGAN项目(https://github.com/Newmu/dcgan_code)。

类似于自编码神经网络,同时把GAN中discriminator和generator组合起来，把DCGAN中的discriminator改造为encoder,把DCGAN中的generator改造为decoder,就形成一个图像翻译网络， 网络结构如下：

![network](https://github.com/lsqpku/img2img/blob/master/doc/network.jpg)

其中discriminator的网络结构代码是：

                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
                return tf.nn.sigmoid(h3), h3

其中卷积层使用了batch norm进行正则化，激活函数是leaky relu，最后进行了sigmoild激活。

generator的网络结构代码是：

                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

                self.h1, self.h1_w, self.h1_b = deconv2d(
                    z, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(
                    h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(
                    h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(
                    h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

                return tf.nn.tanh(h4)

其中卷积层使用了batch norm进行正则化，激活函数是relu，最后进行了tanh激活。



这一结构类似于自编码器，不同的是，在训练这个翻译网络的时候，输入和输出图像属于不同的域。例如在训练一个模糊图像清晰化的翻译网络时，输入训练样本是模糊图像，输出训练样本要尽量和对应的清晰图像接近，因此训练样本必须是每一张模糊样本有一张对应的清晰样本。

示意图：

![legend](https://github.com/lsqpku/img2img/blob/master/doc/LENGEND.JPG)

前面第一副图像就是使用该神经网络对测试样本进行的清晰化处理，使用的是wiki face的彩色男性人脸，使用高斯模糊处理得到模糊图像，这样就得到一大批对偶的模糊－清晰图像训练集（超过3万张）。从清晰化处理效果看，虽然不完美，但清晰化程度大大提高。

那么使用男性人脸训练的模型在对女性人脸进行清晰化处理的效果如何呢？下图是女性测试样本的处理效果（同样对女性样本进行高斯模糊），和男性处理效果基本一致。

(从左到右：原图像、模糊化后的图像和通过模型清晰化的图像）

![image_female](https://github.com/lsqpku/img2img/blob/master/doc/wiki_female.png)

开一下脑洞，对非人脸样本的处理效果如何？下图是对一些非人脸图片的测试效果（同样是先进行了高斯模糊），发现效果也是比较好的，但是色系发生了变化，从冷色调转变为人脸色调。

(从左到右：原图像、模糊化后的图像和通过模型清晰化的图像）

![image_building](https://github.com/lsqpku/img2img/blob/master/doc/wiki_building.png)

把上图转为灰度图像，清晰化效果看起来更明显一些：

(从左到右：原图像、模糊化后的图像和通过模型清晰化的图像）

![image_building](https://github.com/lsqpku/img2img/blob/master/doc/wiki_building_gray.png)

然而，高斯模糊只是模糊方法的一种，使用高斯模糊图像进行训练的模型是否适用于其他模糊方案（例如先resize原图像的1/16,再resize回原尺寸）？下图是对resize模糊方案处理过的测试样本的测试效果：

(左图为对高斯模糊图像进行清晰化后的效果，右图为对resize模糊图像进行清晰化后的效果）

![image_building](https://github.com/lsqpku/img2img/blob/master/doc/wiki_resize_blur.png)

可以看出使用高斯模糊图像训练的模型在处理resize模糊图像效果变差，这是可以理解的，深度学习本质上还是一种模式识别，使用高斯模糊的训练样本，模型会找到高斯模糊的模式。为了使得模型也能够处理resize模糊图像，我们可以把两种样本都作为训练样本进行训练，试验表明对两种情况的清晰化都会比较好，这就是深度神经网络的强大之处，即模型的capacity -- 通过增加测试样本和模型规模，一个模型可以处理更复杂的情况！

上面的模型只是一种神经网络简单的应用，由于模型的损失函数是简单的L2-loss，因此会造成图像模糊化的效果。为了使图像变得更加真实，有人使用GAN进行图像翻译，这里介绍两个比较不错的案例：
1. pix2pix 

article: https://arxiv.org/pdf/1703.10593.pdf　
repo:torch版本https://github.com/phillipi/pix2pix; tensorflow版本：https://github.com/affinelayer/pix2pix-tensorflow）　
这篇文章的核心在于两个：一个是generator的损失函数除了判别真伪以外，加入了L1损失；另一个技巧是在判别真伪时，不是在整个图像范围内判别，而是把图片按patch进行判别，作者称之为patchGAN。经过对比测试发现，在人脸数据上这个模型的效果和上面的基础模型差别不大，但是在facades和citecapes等数据集上，效果要更真实。

2. CycleGan

article:https://arxiv.org/pdf/1703.10593.pdf
repo:https://github.com/hardikbansal/CycleGAN
基础模型以及pix2pix模型要求配对的训练样本，但是实际上有时很难找到大量的此类样本，CycleGan的作者提出了另一种Gan变种, 主要贡献在于只要提供两类数据集即可，不要求严格配对（比如普通马转斑马）。模型较复杂（需要用到2个判别器和两个生成器），感兴趣的可参阅https://hardikbansal.github.io/CycleGANBlog/

总的来说，使用神经网络进行图像翻译，简单高效，结合Gan网络让还原后的图像更加逼真，大家可以尝试在更多场景应用图像翻译的思路，发挥深度学习的威力。
