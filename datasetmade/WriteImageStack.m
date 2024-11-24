function WriteImageStack(img,name,nameImg)
newname1 =strcat(name,nameImg);
n = size(img, 3);
for i = 1 : n
    imwrite(uint8(img(:, :, i)), newname1, 'WriteMode', 'append');
end

end

