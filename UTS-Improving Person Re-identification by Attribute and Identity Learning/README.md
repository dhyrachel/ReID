论文：《Improving Person Re-identification by Attribute and Identity Learning》
算法原理：
 
构建APR网络实现属性+ID的reid行人重识别：
首先以ResNet-50为基准网络提取图像特征，整合到pool层后，在计算loss前有M+1个全连接层，一个id分类的loss，M个属性识别的losses，M是属性的个数。新的全连接层命名为FC0，FC1，...，FCM ，FC0 用于id分类，FC0，FC1，...，FCM 用于属性分类。

源代码在matlab上实现，具体的运行环境为：
Matlab2018a+VS2017+CUDA9.0+cudnn7.0+MatconvNet

以market1501为数据集时：
利用源代码中训练好的模型进行属性预测，属性的平均预测准确率为：
precision1 =0.8645
precision2 =0.8708
precision3 =0.8365
precision4 =0.9366
precision5 =0.9332
precision6 =0.9146
precision7 =0.8279
precision8 =0.8998
precision9 = 0.7507
precision10=0.0287（该项预测准确率很低，确切的预测属性为‘是否戴帽子’）         
precision11=0.7340
precision12 =0.6991
average =0.7747
在论文中，APR在market1501上的属性识别准确度为：0.8533，与我们的实验结果0.7747有差异。
 
该差异跟属性‘是否戴帽子’的低识别正确率有关~~
这跟作者原文中的描述一致：“我们发现APR网络使一些属性的识别率降低了，比如DukeMTMC-reID中的“性别”和“靴子”。然而图九中展示了这些属性在提升re-ID表现中很重要。原因可能在于APR的多任务天性。因为这一模型是为了re-ID优化的（图七），某些属性的模棱两可的图片可能预测错误。”所以由于某些属性的识别难度较大，加入了属性识别的reid可能是把双刃剑。

而其mAP和CMC等评价指标需通过相关评价指标工具来测定。
由评价指标工具计算得：
mAP
single query:                             mAP = 0.646712, r1 precision = 0.842933
average of confusion matrix with single query:  mAP = 0.534423, r1 precision = 0.609460
该结果与论文中声明的结果一致
 
摄像头相关矩阵：
mAP
 
R1
 
CMC：
 

关于属性行人重识别方法的改进建议：
1.	行人属性受季节影响，而属性需要手动设计，属性需要根据季节来设计，因为不同季节人们的衣着风格不同，所以需要分开来设计，增重了手工设计的负担，所以最好一开始就分开设计，并训练适合不同季节的模型
2.	不同属性对分类的增益不同，甚至可能出现降低分类准确度的可能，所以属性的选择和属性个数的选择是个难以确定的因素，也在不同环境中受到影响，所以可以先做一个只有几个主要属性的模型，而后根据特定情形需要增加对应的属性


