

clc; clear; close all;


Jaccard_all = [];
dice_all = [];
precision_all = [];
recall_all = [];

namelist_pred = dir('G:\zhanghong\test_output\*.tif');
namelist_label = dir('G:\zhanghong\test_lable\*.tif');

saveDir = '.\IoU\';  %  ��ʾoverlap�Ľ��
mkdir(saveDir);
%maskDir  = '.\mask';         % ԭʼͼ��ǩͼ��·��

num = length(namelist_label);   % �ܹ���ͼ�����
for f = 1:num
    save_name = [num2str(f, '%04d'), '_pred.png'];
    pred_name = 'G:\zhanghong\test_output';
    img_name = 'G:\zhanghong\test_lable';
    
    
    %I = logical(ReadTiff(fullfile(img_name,namelist_label(f).name)));      %��ȡԭʼ��ǩͼ,������ж�ֵ������
    I = logical(imread(fullfile(img_name,namelist_label(f).name)));
    prob = imread(fullfile(pred_name,namelist_pred(f).name));  %��ȡԤ��ͼƬ
    prob_map = logical(prob);
    
%     pred_name = [num2str(f, '%d'), '_pred.png'];
%     prob = imread(fullfile(maskDir,pred_name));  %��ȡԤ��ͼƬ
% %     ����ֵ������ȡԤ��ͼƬ��ֵͼ������70��Ϊ1��С��128��Ϊ0
%     prob_map = double(prob);
%     tIm = prob_map>128;
%     prob_map = logical(tIm);
    
    %% ����Jaccard��dice Iou
    
    jr = double(sum(uint8(prob_map(:) & I(:)))) / double(sum(uint8(prob_map(:) | I(:))));
    Jaccard_all=[Jaccard_all jr];
    
    % dice ratio
    dr = 2*double(sum(uint8(prob_map(:) & I(:)))) / double(sum(uint8(prob_map(:))) + sum(uint8(I(:))));
    dice_all=[dice_all dr];
    
    %% ����precision��reacll
 
    pr = double(sum(uint8(prob_map(:) & I(:)))) / double(sum(uint8(prob_map(:)))); 
    precision_all=[precision_all pr];
    
    re = double(sum(uint8(prob_map(:) & I(:)))) / double(sum(uint8(I(:))));  
    recall_all=[recall_all re];
    
    % ��ʾoverlap�Ľ�� ��ɫ��ʾ©��FN ��ɫ��ʾ����TP ��ɫ��ʾ���FP
    img2(:,:,1) = uint8(I - (prob_map & I)).*255;   % R ©��FN
    img2(:,:,2) = uint8(prob_map & I).*255;         % G ���� FP
    img2(:,:,3) = uint8((prob_map | I) - I).*255;   % B ���FP
    imwrite(img2,fullfile(saveDir,save_name));
end

%����ƽ��
Jaccard_mean = mean(Jaccard_all)
dice_mean = mean(dice_all)
precision_mean = mean(precision_all)
recall_mean = mean(recall_all)

