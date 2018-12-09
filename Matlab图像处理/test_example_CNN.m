function test_example_CNN



train_x = load('D:\所有程序\CNN\train_xx');
train_y= load('D:\所有程序\CNN\train_yy');
test_x = load('D:\所有程序\CNN\test_xx');
test_y = load('D:\所有程序\CNN\test_yy');

train_x = double(reshape(train_x',28,28,3000))/255;
test_x = double(reshape(test_x',28,28,1500))/255;
train_y = double(train_y');
test_y = double(test_y');  


rand('state',0)


cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 3) %convolution layer
};


opts.alpha = 1;
opts.batchsize = 10;
opts.numepochs = 3;

cnn = cnnsetup(cnn, train_x, train_y);  

cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);

% plot mean squared error
figure; plot(cnn.rL);xlabel('X (训练次数)','FontSize',12,'FontWeight','bold','Color','k');ylabel('Y (均方误差)','FontSize',12,'FontWeight','bold','Color','k')
assert(er<0.12, 'Too big error');
