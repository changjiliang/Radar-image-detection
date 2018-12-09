
%提取数据，船舶样本转化成3000个1*784的矩阵
file_path0 = 'C:\Users\cjl\Desktop\Matlab工具箱、训练样本\train_0\';
img_path_list0 = dir(strcat(file_path0,'*.png'));
img_num0 = length(img_path_list0);
for j = 1:img_num0
    image0_name = img_path_list0(j).name;
    image0 =  imread(strcat(file_path0,image0_name));
    T0 = reshape(image0,1,784);
    image0_new{j} = T0;
end

%将3000个1*784的矩阵列向拼接在一起
i0=2;
train_x0 = image0_new{1};
while i0<=img_num0
    train_x0 = [train_x0 ; image0_new{i0}];
    i0=i0+1;
end

%提取数据，非船舶样本转化成1000个1*784的矩阵
file_path1 ='C:\Users\cjl\Desktop\Matlab工具箱、训练样本\train_1\';
img_path_list1 = dir(strcat(file_path1,'*.png'));
img_num1 = length(img_path_list1);
for k = 1:img_num1
  image1_name = img_path_list1(k).name;  
  image1 =  imread(strcat(file_path1,image1_name));
   T1= reshape(image1,1,784);
    image1_new{k} = T1;
end
%将1500个1*784的矩阵列向拼接在一起
i1=2;
train_x1 = image1_new{1};
while i1<=img_num1
    train_x1 = [train_x1 ; image1_new{i1}];
    i1=(i1+1);
end
%将3000个船舶样本和1500个非船舶样本列向拼接
train_x = [train_x0;train_x1];


