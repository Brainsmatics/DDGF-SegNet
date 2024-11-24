

clc; clear; close all;


Jaccard_all = [];
dice_all = [];
precision_all = [];
recall_all = [];

namelist_pred = dir('G:\zhanghong\test_output\*.tif');
namelist_label = dir('G:\zhanghong\test_lable\*.tif');

saveDir = '.\IoU\';  %  显示overlap的结果
mkdir(saveDir);
%maskDir  = '.\mask';         % 原始图标签图的路径

num = length(namelist_label);   % 总共的图像个数
for f = 1:num
    save_name = [num2str(f, '%04d'), '_pred.png'];
    pred_name = 'G:\zhanghong\test_output';
    img_name = 'G:\zhanghong\test_lable';
    
    
    %I = logical(ReadTiff(fullfile(img_name,namelist_label(f).name)));      %读取原始标签图,将其进行二值化处理
    I = logical(imread(fullfile(img_name,namelist_label(f).name)));
    prob = imread(fullfile(pred_name,namelist_pred(f).name));  %读取预测图片
    prob_map = logical(prob);
    
%     pred_name = [num2str(f, '%d'), '_pred.png'];
%     prob = imread(fullfile(maskDir,pred_name));  %读取预测图片
% %     用阈值方法获取预测图片二值图，大于70的为1，小于128的为0
%     prob_map = double(prob);
%     tIm = prob_map>128;
%     prob_map = logical(tIm);
    
    %% 计算Jaccard和dice Iou
    
    jr = double(sum(uint8(prob_map(:) & I(:)))) / double(sum(uint8(prob_map(:) | I(:))));
    Jaccard_all=[Jaccard_all jr];
    
    % dice ratio
    dr = 2*double(sum(uint8(prob_map(:) & I(:)))) / double(sum(uint8(prob_map(:))) + sum(uint8(I(:))));
    dice_all=[dice_all dr];
    
    %% 计算precision和reacll
 
    pr = double(sum(uint8(prob_map(:) & I(:)))) / double(sum(uint8(prob_map(:)))); 
    precision_all=[precision_all pr];
    
    re = double(sum(uint8(prob_map(:) & I(:)))) / double(sum(uint8(I(:))));  
    recall_all=[recall_all re];
    
    % 显示overlap的结果 红色表示漏检FN 绿色表示正检TP 蓝色表示错检FP
    img2(:,:,1) = uint8(I - (prob_map & I)).*255;   % R 漏检FN
    img2(:,:,2) = uint8(prob_map & I).*255;         % G 正检 FP
    img2(:,:,3) = uint8((prob_map | I) - I).*255;   % B 错检FP
    imwrite(img2,fullfile(saveDir,save_name));
end

%计算平均
Jaccard_mean = mean(Jaccard_all)
dice_mean = mean(dice_all)
precision_mean = mean(precision_all)
recall_mean = mean(recall_all)

