% In this file, we densely extract the feature
% It's similar to the 10-crop in the ResNet Paper.
clear;
netStruct = load('F:\jinhao\ReID\Code\reID_attri\data\resnet_market\net-epoch-55.mat');
%--------add norm
net = dagnn.DagNN.loadobj(netStruct.net);
net.addLayer('lrn_test',dagnn.LRN('param',[2048,0,1,0.5]),{'feature'},{'featured'},{});
clear netStruct;
net.mode = 'test' ;
net.move('gpu') ;
net.conserveMemory = true;
im_mean = net.meta(1).normalization.averageImage;
im_mean = imresize(im_mean,[256,256]);
p = dir('F:\jinhao\ReID\Code\reID_attri\dataset\Market-1501_Attribute\Market-1501\bounding_box_test\*jpg');
ff = [];
%%------------------------------gallery
for i = 1:200:numel(p)
    disp(i);
    oim = [];
    str=[];
    for j=1:min(200,numel(p)-i+1)
        str = strcat('F:\jinhao\ReID\Code\reID_attri\dataset\Market-1501_Attribute\Market-1501\bounding_box_test\',p(i+j-1).name);
        imt = imresize(imread(str),[256,256]);
        oim = cat(4,oim,imt);
    end
    f = getFeature2(net,oim,im_mean,'data','pool5d');
    f = sum(sum(f,1),2);
    f2 = getFeature2(net,fliplr(oim),im_mean,'data','pool5d');
    f2 = sum(sum(f2,1),2);
    f = f+f2;
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    s = sqrt(sum(f.^2,2));
    dim = size(f,2);
    s = repmat(s,1,dim);
    f = f./s;
    ff = cat(1,ff,f);
end
save('resnet_gallery.mat','ff','-v7.3');
%---------query
p = dir('F:\jinhao\ReID\Code\reID_attri\dataset\Market-1501_Attribute\Market-1501\query\*jpg');
ff = [];
for i = 1:200:numel(p)
    disp(i);
    oim = [];
    str=[];
    for j=1:min(200,numel(p)-i+1)
        str = strcat('F:\jinhao\ReID\Code\reID_attri\dataset\Market-1501_Attribute\Market-1501\query\',p(i+j-1).name);
        imt = imresize(imread(str),[256,256]);
        oim = cat(4,oim,imt);
    end
    f = getFeature2(net,oim,im_mean,'data','pool5d');
    f = sum(sum(f,1),2);
    f2 = getFeature2(net,fliplr(oim),im_mean,'data','pool5d');
    f2 = sum(sum(f2,1),2);
    f = f+f2;
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    s = sqrt(sum(f.^2,2));
    dim = size(f,2);
    s = repmat(s,1,dim);
    f = f./s;
    ff = cat(1,ff,f);
end
save('resnet_query.mat','ff','-v7.3');

