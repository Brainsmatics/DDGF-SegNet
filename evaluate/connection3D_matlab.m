

%% ɾ��С��ͨ��õ��Ľ��
clc; clear all;
mask = zeros(1360,1360,10);  % ͼ��ĳ�������

num = 10;
for i =1:num   % ͼ�������
    mask(:,:,i)  = imread(['.\pred\',num2str(i,'%04d'),'.png']);    
end

% ����������ͨ��ɾ��С��100��������ͨ����
%BW =mask>210;   % ��ֵ�ָ�   �����޸�
%L = bwlabeln(BW, 6); %������ͨ��
%S = regionprops(L, 'Area');
%bw2 = ismember(L, find([S.Area] >= 200));  % ��������100����ͨ����   �����޸�

% �ٴμ�����ͨ��
tic
imgs = bwlabeln(mask, 6); %������ͨ��
toc

for i =1:num      
    seg = uint8(imgs(:,:,i));
    imwrite(seg,'seg.tif','tiff','WriteMode','append'); 
end





