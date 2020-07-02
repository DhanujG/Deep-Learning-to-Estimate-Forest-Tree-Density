%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File to create grount truth density map for test set%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc; clear all;
dataset = 'T';
dataset_name = ['shanghaitech_part_' dataset ];
path = ['C:\Users\dhanu\Documents\School\442\Project\data\original\trees\train_data\images\'];
%path = ['/mnt/c/Users/dhanu/Documents/School/442/Project/data/custom_test'];
%gt_path = ['/mnt/c/Users/dhanu/Documents/School/442/Project/data/original/shanghaitech/part_' dataset '_final/test_data/ground_truth/'];
%gt_path_csv = ['C:\Users\dhanu\Documents\School\442\Project\data\original\ground_trees_cvs\'];
gt_path_csv = ['C:\Users\dhanu\Documents\School\442\Project\PCC_net\data\trees\train_data\den\'];
gt_path = 'C:\Users\dhanu\Documents\School\442\Project\data\original\trees\train_data\ground_truth\';

mkdir(gt_path_csv )
if (dataset == 'T')
    num_images = 45;
else
    num_images = 316;
end

for i = 1:num_images    
    if (mod(i,10)==0)
        fprintf(1,'Processing %3d/%d files\n', i, num_images);
    end
    (strcat(gt_path, 'GT_IMG_',num2str(i),'.mat'));
    load(strcat(gt_path, 'GT_IMG_',num2str(i),'.mat'));
    input_img_name = strcat(path,'IMG_',num2str(i),'.jpg');
    im = imread(input_img_name);
    [h, w, c] = size(im);
    if (c == 3)
        im = rgb2gray(im);
    end     
    annPoints =  image_info{1}.location;   
    [h, w, c] = size(im);
    im_density = get_density_map_gaussian(im,annPoints);    
    %csvwrite([gt_path_csv ,'IMG_',num2str(i) '.csv'], im_density);
    csvwrite([gt_path_csv ,'',num2str(i) '.csv'], im_density);
    %imwrite(im,[ num2str(i) '.jpg']); 
end