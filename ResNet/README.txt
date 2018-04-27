
参考：https://blog.csdn.net/liangyihuai/article/details/79140481

1. 训练loss下降，但是eval没有下降，同时对train的你和也同样有问题
    打印：step = 1, dev loss = 1.7926e+00, dev accuracy = 0.1667
    其中，accuracy始终不变

2. 2018-04-27 loss到200布左右开始下降，在1000步左右到达85%交叉验证准确率