
%����һ��3000*1�ľ���
train_y0 = ones(3000,1);
train_y1 = zeros(3000,1);
train_y2 = [train_y0,train_y1];
%����һ��1500*1�ľ���
train_y3 = zeros(1500,1);
train_y4 = ones(1500,1);
train_y5 = [train_y3,train_y4];
%����������ƴ��
train_y = [train_y2;train_y5];