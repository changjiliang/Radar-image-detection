file_path = 'D:\雷达图\';
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
    J3 = medfilt2(J1,[15 15]);
    J4 = medfilt2(J2,[15 15]);
    J3 = im2bw(J3,0);
    J4 = im2bw(J4,0);
    J5 = bwareaopen(J3,3000);
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
    %imwrite(J9,['D:\所有程序\结果图片（试验）\对比\',num2str(j),'.png']);
    figure(j);
    imshow(J9);
    I1 = regionprops(J7,'area','boundingbox','Centroid');
    Sta{j} = I1;
    for i1 = 1:length(Sta{j})
        floor(Sta{j}(i1).BoundingBox(1));
        floor(Sta{j}(i1).BoundingBox(2));
        if Sta{j}(i1).BoundingBox(4) >= Sta{j}(i1).BoundingBox(3)
            Sta{j}(i1).BoundingBox(4) = Sta{j}(i1).BoundingBox(4)+10;
            Sta{j}(i1).BoundingBox(3) = Sta{j}(i1).BoundingBox(4);
        else
            Sta{j}(i1).BoundingBox(3) = Sta{j}(i1).BoundingBox(3)+10;
            Sta{j}(i1).BoundingBox(4) = Sta{j}(i1).BoundingBox(3);
        end
    end



   for i1 = 1:length(Sta{j})
        for i = 1:size(Sta{j}(i1).BoundingBox,1)
            rectangle('position',[floor(Sta{j}(i1).BoundingBox(1)-10),floor(Sta{j}(i1).BoundingBox(2)-10),Sta{j}(i1).BoundingBox(3)+10,Sta{j}(i1).BoundingBox(4)+10],'EdgeColor','r');        %%画框
            text(floor(Sta{j}(i1).BoundingBox(1)),floor(Sta{j}(i1).BoundingBox(2)),num2str(i1),'color','green');
            saveas(gcf,['D:\所有程序\结果图片（试验）\对比1\',num2str(j),'.png'])
        end
   end


% saveas(gcf,[num2str(j),'.png']);
% test_x = [length(Sta{j}),784]; %%%初始化 test_x
% 
% for k = 1:length(Sta{j})
%     
%     I3 = imcrop(J9,[Sta{j}(k).BoundingBox(1)-10, Sta{j}(k).BoundingBox(2)-10,   Sta{j}(k).BoundingBox(3)+10,   Sta{j}(k).BoundingBox(4)+10]);
%     I4 = imresize(I3,[28,28]);
% %     figure(k);imshow(I4);
%     I5 = reshape(I4,1,784);
%     test_x(k,1:784) = I5(1,:);
% end
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% net = load('D:\所有程序\CNN\NET');
% 
% test_x = double(reshape(test_x',28,28,length(Sta{j})));
% NET = cnnff(net, test_x);
% 
% figure(j);imshow(J7);
% 
% for i1 = 1:length(Sta{j})
%     
%      if NET.net.o(1,i1)>NET.net.o(2,i1)
%         rectangle('position',[floor(Sta{j}(i1).BoundingBox(1)-10),floor(Sta{j}(i1).BoundingBox(2)-10),Sta{j}(i1).BoundingBox(3)+10,Sta{j}(i1).BoundingBox(4)+10],'EdgeColor','r');
%         text(floor(Sta{j}(i1).BoundingBox(1)),floor(Sta{j}(i1).BoundingBox(2)),'ship','color','green');
%     else
%         rectangle('position',[floor(Sta{j}(i1).BoundingBox(1)-10),floor(Sta{j}(i1).BoundingBox(2)-10),Sta{j}(i1).BoundingBox(3)+10,Sta{j}(i1).BoundingBox(4)+10],'EdgeColor','g');
%         text(floor(Sta{j}(i1).BoundingBox(1)),floor(Sta{j}(i1).BoundingBox(2)),'f-ship','color','red');
%     end
%     
%     saveas(gcf,['D:\所有程序\结果图片（试验）\测试1\',num2str(j),'.png']);
% end
% end


end    