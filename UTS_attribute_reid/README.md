# Improving Person Re-identification by Attribute and Identity Learning
Created by Yutian Lin, on March 30, 2017.

In this package, we provide our training and testing code for the paper. 
All codes have been test on Ubuntu14.04 with Matlab R2015b.
We take Market1501 as an example.

#Datasets
Market1501 (You can find it on https://liangzheng.org)
Market1501-attribute (You can find it on https://github.com/vana77/Market-1501_Attribute)

#To train
1.Install and compile matconvnet. Replace cnn_train_dag.m with the one in the shared file.
If you meet something wrong, you can refer to http://www.vlfeat.org/matconvnet/install/

2.Download above datasets, and add your dataset path into `pre_market.m` and run it. 

3.Run `train_market.m` to have fun.

#To test
1.Install and compile matconvnet. 

2.For attribute recognition:
(1)Run `pre_market_test.m` to generate ground truth attribute result.
(2)Run `predict_atteibute_market.m` to predict attributes and calculate accuracy.

3.For person re-ID:
(1)Run `test/test_res_market.m` to extract the features of gallery and query. They will store in a .mat file. You can use it to do evaluation.

