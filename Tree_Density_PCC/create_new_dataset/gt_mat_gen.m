%Changes the resolution of image files to 1024*768 to make them compatible
%with ShanghaiTech dataset part_B 

%Generates ground truth .mat files for each image file using head
%annotations

clear all
clc
close all
filePattern = fullfile('images', '\*.jpg');
ImageFiles = dir(filePattern);
n = length(ImageFiles)
read_path = 'images\';
store_path = 'ground-truth\';
t = 0;                          %number of files initially in training set

for i=1:20
    im = imread([read_path 'IMG_' num2str(i+t) '.jpg']);
    
    im = imresize(im, [512 680]);
    imwrite(im,[read_path 'IMG_' num2str(i+t) '.jpg'], 'jpg'); 
    figure
    imshow(im)
    [x,y] = getpts;
    image_info{1,1}.location = [x y];
    image_info{1,1}.number = size(x,1);
    save([store_path 'GT_IMG_' num2str(t+i) '.mat'], 'image_info')
    
    im = imread([read_path 'IMG_' num2str(i+t) '.jpg']);
    im = imrotate(im,-180,'bilinear','crop');
    im = flip(im ,2);
    im = imresize(im, [512 680]);
    imwrite(im,[read_path 'IMG_' num2str(i+20+t) '.jpg'], 'jpg'); 
    figure
    imshow(im)
    [x,y] = getpts;
    image_info{1,1}.location = [x y];
    image_info{1,1}.number = size(x,1);
    save([store_path 'GT_IMG_' num2str(t+i+20) '.mat'], 'image_info')
    
    im = imread([read_path 'IMG_' num2str(i+t) '.jpg']);
    im = flip(im ,2);
    im = imresize(im, [512 680]);
    imwrite(im,[read_path 'IMG_' num2str(i+40+t) '.jpg'], 'jpg'); 
    figure
    imshow(im)
    [x,y] = getpts;
    image_info{1,1}.location = [x y];
    image_info{1,1}.number = size(x,1);
    save([store_path 'GT_IMG_' num2str(t+i+40) '.mat'], 'image_info')
    
    close
end