file_path = 'D:\长江雷达挑选图\';
img_path_list = dir(strcat(file_path,'*.png'));
img_num = length(img_path_list);
for j = 1:img_num
     image_name1 = img_path_list(j).name;
     image_name2 = img_path_list(j+5).name;
     image1 =  imread(strcat(file_path,image_name1));
     image2 =  imread(strcat(file_path,image_name2));
     image3 = rgb2gray(image1);
     image4 = rgb2gray(image2);
     J1 = image3;
J2 = image4;
L = size(J1, 1);
W = size(J1, 2);
for y=1:L
    for x=1:W
        if(y+(1568/2048)*x-2048<0)
            J1(y,x)=0;
            J2(y,x)=0;
        else
            J1(y,x)=J1(y,x);
            J2(y,x)=J2(y,x);
        end
        if(y+(998/1293)*x-2630.7>0)
            J1(y,x)=0;
            J2(y,x)=0;
        else
            J1(y,x)=J1(y,x);
            J2(y,x)=J2(y,x);
        end
        
    end
end
J3 = medfilt2(J1,[15 15 ]);%%中值滤波
J4 = medfilt2(J2,[15 15 ]);
% J3 = im2bw(J3,0);%%二值化
% J4 = im2bw(J4,0);
J5 = bwareaopen(J3,3000);%%去除不合格联通与
J6 = bwareaopen(J4,3000);
xx = zeros(L,W);
yy = zeros(L,W);
ww = zeros(L,W);
zz = zeros(L,W);
uu = xx;
vv = xx;
for a = 1:W
    for b = 1:L
        xx(a,b) = J3(a,b);
        yy(a,b) = J5(a,b);
        ww(a,b) = J4(a,b);
        zz(a,b) = J6(a,b);
    end
end
J7 = abs(yy-xx);
J8 = abs(ww-zz);
J7 = bwareaopen(J7,300);
J8 = bwareaopen(J8,300);
for c = 1:W
    for d = 1:L
        uu(c,d) = J7(c,d);
        vv(c,d) = J8(c,d);
    end
end
J9 = abs(uu-vv);
figure(j);
% imshow(J9);
imwrite(J9,['D:\所有程序\结果图片（试验）\帧差图\',num2str(j),'.png']);
end