
%��ȡ���ݣ���������ת����3000��1*784�ľ���
file_path0 = 'C:\Users\cjl\Desktop\Matlab�����䡢ѵ������\train_0\';
img_path_list0 = dir(strcat(file_path0,'*.png'));
img_num0 = length(img_path_list0);
for j = 1:img_num0
    image0_name = img_path_list0(j).name;
    image0 =  imread(strcat(file_path0,image0_name));
    T0 = reshape(image0,1,784);
    image0_new{j} = T0;
end

%��3000��1*784�ľ�������ƴ����һ��
i0=2;
train_x0 = image0_new{1};
while i0<=img_num0
    train_x0 = [train_x0 ; image0_new{i0}];
    i0=i0+1;
end

%��ȡ���ݣ��Ǵ�������ת����1000��1*784�ľ���
file_path1 ='C:\Users\cjl\Desktop\Matlab�����䡢ѵ������\train_1\';
img_path_list1 = dir(strcat(file_path1,'*.png'));
img_num1 = length(img_path_list1);
for k = 1:img_num1
  image1_name = img_path_list1(k).name;  
  image1 =  imread(strcat(file_path1,image1_name));
   T1= reshape(image1,1,784);
    image1_new{k} = T1;
end
%��1500��1*784�ľ�������ƴ����һ��
i1=2;
train_x1 = image1_new{1};
while i1<=img_num1
    train_x1 = [train_x1 ; image1_new{i1}];
    i1=(i1+1);
end
%��3000������������1500���Ǵ�����������ƴ��
train_x = [train_x0;train_x1];


