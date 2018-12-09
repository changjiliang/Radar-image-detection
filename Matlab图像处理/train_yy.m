
%创建一个3000*1的矩阵
train_y0 = ones(3000,1);
train_y1 = zeros(3000,1);
train_y2 = [train_y0,train_y1];
%创建一个1500*1的矩阵
train_y3 = zeros(1500,1);
train_y4 = ones(1500,1);
train_y5 = [train_y3,train_y4];
%将它们列向拼接
train_y = [train_y2;train_y5];