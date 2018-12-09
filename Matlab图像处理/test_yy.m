%创建一个1500*1的矩阵
test_y0 = ones(1000,1);
test_y1 = zeros(1000,1);
test_y2 = [test_y0,test_y1];
%创建一个500*1的矩阵
test_y3 = zeros(500,1);
test_y4 = ones(500,1);
test_y5 = [test_y3,test_y4];

%将它们列向拼接
test_y = [test_y2;test_y5];