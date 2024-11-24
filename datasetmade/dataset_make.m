clc; clear; close all;
namelist1 = '.\220374\transfer\merge_220374_1.tif';
namelist_label1 = '.\220374\transfer\merge_220374_1.Labels.smooth.tif';

namelist2 = '.\220374\transfer\merge_220374_2.tif';
namelist_label2 = '.\220374\transfer\merge_220374_2.Labels.smooth.tif';

namelist3 = '.\221000\221000.1.tif';
namelist_label3 = '.\221000\221000.1.Labels.tif';

namelist4 = '.\221000\221000.2.tif';
namelist_label4 = '.\221000\221000.2.Labels.tif';
%% 
a = ReadTiff(namelist1);
a1 = ReadTiff(namelist_label1);
[y1,x1,z1] = size(a);

b = ReadTiff(namelist2);
b1 = ReadTiff(namelist_label2);
[y2,x2,z2] = size(b);

c = ReadTiff(namelist3);
c1 = ReadTiff(namelist_label3);
[y3,x3,z3] = size(c);

d = ReadTiff(namelist4);
d1 = ReadTiff(namelist_label4);
[y4,x4,z4] = size(d);
%% train
for i = 1 :120
    fp=fopen('train.txt','a');
    choose_img = a(:,1:512,i);
    namenum=num2str(i,'%04d');
    img_name =strcat('a',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = a1(:,1:512,i);
    label_name =strcat('a_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
  %a
    choose_img = a(:,513:1024,i);
    namenum=num2str(i+150,'%04d');
    img_name =strcat('a',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = a1(:,513:1024,i);
    label_name =strcat('a_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
 %b
    choose_img = b(:,1:512,i);
    namenum=num2str(i,'%04d');
    img_name =strcat('b',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = b1(:,1:512,i);
    label_name =strcat('b_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
  %b
    choose_img = b(:,513:1024,i);
    namenum=num2str(i+150,'%04d');
    img_name =strcat('b',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = b1(:,513:1024,i);
    label_name =strcat('b_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
 %c
    choose_img = c(:,1:512,i);
    namenum=num2str(i,'%04d');
    img_name =strcat('c',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = c1(:,1:512,i);
    label_name =strcat('c_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
  %c
    choose_img = c(:,513:1024,i);
    namenum=num2str(i+150,'%04d');
    img_name =strcat('c',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = c1(:,513:1024,i);
    label_name =strcat('c_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
 %d
    choose_img = d(:,1:512,i);
    namenum=num2str(i,'%04d');
    img_name =strcat('d',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = d1(:,1:512,i);
    label_name =strcat('d_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
 %d
    choose_img = d(:,513:1024,i);
    namenum=num2str(i+150,'%04d');
    img_name =strcat('d',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = d1(:,513:1024,i);
    label_name =strcat('d_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
    
    fclose(fp);
end
%% val
for i = 121 :150
    fp=fopen('val.txt','a');
    choose_img = a(:,1:512,i);
    namenum=num2str(i,'%04d');
    img_name =strcat('a',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = a1(:,1:512,i);
    label_name =strcat('a_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
  %a
    choose_img = a(:,513:1024,i);
    namenum=num2str(i+150,'%04d');
    img_name =strcat('a',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = a1(:,513:1024,i);
    label_name =strcat('a_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
 %b
    choose_img = b(:,1:512,i);
    namenum=num2str(i,'%04d');
    img_name =strcat('b',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = b1(:,1:512,i);
    label_name =strcat('b_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
  %b
    choose_img = b(:,513:1024,i);
    namenum=num2str(i+150,'%04d');
    img_name =strcat('b',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = b1(:,513:1024,i);
    label_name =strcat('b_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
 %c
    choose_img = c(:,1:512,i);
    namenum=num2str(i,'%04d');
    img_name =strcat('c',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = c1(:,1:512,i);
    label_name =strcat('c_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
  %c
    choose_img = c(:,513:1024,i);
    namenum=num2str(i+150,'%04d');
    img_name =strcat('c',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = c1(:,513:1024,i);
    label_name =strcat('c_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
 %d
    choose_img = d(:,1:512,i);
    namenum=num2str(i,'%04d');
    img_name =strcat('d',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = d1(:,1:512,i);
    label_name =strcat('d_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
 %d
    choose_img = d(:,513:1024,i);
    namenum=num2str(i+150,'%04d');
    img_name =strcat('d',namenum,'.tif');
    WriteImageStack(choose_img,'./imgs/',img_name);
    
    choose_label = d1(:,513:1024,i);
    label_name =strcat('d_labels_',namenum,'.tif');
    WriteImageStack(choose_label,'./labels/',label_name);
    fprintf(fp,'%s,%s\r\n',img_name,label_name);
    
    fclose(fp);
end