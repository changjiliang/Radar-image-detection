
%��ȡ���ݣ���������ת����1500��1*784�ľ���
file_path2 ='C:\Users\cjl\Desktop\Matlab�����䡢ѵ������\test_0\';
img_path_list2 = dir(strcat(file_path2,'*.png'));
img_num2 = length(img_path_list2);
for c = 1:img_num2
    image2_name = img_path_list2(c).name;
    image2 =  imread(strcat(file_path2,image2_name));
    T2 = reshape(image2,1,784);
    image2_new{c} = T2;
end

%��1500��1*784�ľ�������ƴ����һ��
i2=2;
test_x2 = image2_new{1};
while i2<=img_num2
    test_x2 = [test_x2 ; image2_new{i2}];
    i2=i2+1;
end

%��ȡ���ݣ��Ǵ�������ת����500��1*784�ľ���
file_path3 = 'C:\Users\cjl\Desktop\Matlab�����䡢ѵ������\test_1\';
img_path_list3 = dir(strcat(file_path3,'*.png'));
img_num3 = length(img_path_list3);
for b = 1:img_num3
    image3_name = img_path_list3(b).name;  
    image3 =  imread(strcat(file_path3,image3_name));
    T3= reshape(image3,1,784);
    image3_new{b} = T3;
end

%��500��1*784�ľ�������ƴ����һ��
i3=2;
test_x3 = image3_new{1};
while i3<=img_num3
 test_x3 = [test_x3 ; image3_new{i3}];
    i3=(i3+1);
end
%��1500������������500���Ǵ�����������ƴ��
test_x = [test_x2;test_x3];

