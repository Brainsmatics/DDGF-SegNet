

%% 删除小连通域得到的结果
clc; clear all;
mask = zeros(1360,1360,10);  % 图像的长宽，张数

num = 10;
for i =1:num   % 图像的张数
    mask(:,:,i)  = imread(['.\pred\',num2str(i,'%04d'),'.png']);    
end

% 计算立体连通域并删除小于100的立体连通区域
%BW =mask>210;   % 阈值分割   可以修改
%L = bwlabeln(BW, 6); %计算连通域
%S = regionprops(L, 'Area');
%bw2 = ismember(L, find([S.Area] >= 200));  % 保留大于100的连通区域   可以修改

% 再次计算连通域
tic
imgs = bwlabeln(mask, 6); %计算连通域
toc

for i =1:num      
    seg = uint8(imgs(:,:,i));
    imwrite(seg,'seg.tif','tiff','WriteMode','append'); 
end





