import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

y = [0.6865,0.8030,0.9274,0.9357,0.9465,0.9507,0.9556,0.9665,0.9674,0.9774,0.9813,0.9824,0.9891,0.9856,0.9885,0.9911,0.9870,0.9894,0.9933,0.9950,0.9915,0.9963,0.9943,0.9959,0.9989,0.9987,0.9950,0.9956,0.9978,0.9980,0.9950,0.9920]



b = [0.7717,0.8183,0.8867,0.9083,0.9050,0.9050,0.9083,0.9117,0.9133,0.8900,0.9117,0.9133,0.9100,0.9117,0.9217,0.9067,0.9117,0.9100,0.9067,0.9117,0.9083,0.9133,0.9000,0.9183,0.9117,0.9083,0.9083,0.9250,0.9217,0.9133,0.9217,0.9283]
plt.plot(x,y,'--+')
plt.plot(x,b,'--+')
plt.ylim(0.65,1)
xmajorLocator = MultipleLocator(2)
my_x_ticks = np.arange(0,30,1)
plt.xticks(my_x_ticks)
plt.xticks([0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
           ['1','3','5','7','9','11','13','15','17','19','21',
            '23','25','27','29','31'])
plt.title('model-accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.savefig('./zhunquelv.png')
plt.show()
