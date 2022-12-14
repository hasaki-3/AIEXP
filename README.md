# AIEXP🏫

一、运行方法

1、运行Train.py进行训练，自定轮数保存模型。

2、运行Test.py并加载训练好的模型进行测试，建议训练轮数大于30轮。

二、效果提升

使用transform扩充数据集进行效果提升，并增加训练轮数至50轮，在Test.py中测试效果提升，置信度从0.83提升至1.00。

三、文件说明

read_data.py：数据读取，进行数据增强，数据加载并转为pytorch可用的数据

GenAnchors.py：针对每个像素点生成锚框，确定偏移量和中心点

GenBoxes.py：计算两个锚框或边界框列表中成对的交并比，将最接近的真实边界框分配给锚框，使用真实边界框标记锚框：如果一个锚框没有被分配，我们标记其为背景（值为零）

Trans.py: 定义中心+宽度表示锚框与左上角和右下角表示锚框，进行两种方式间的转换

nms.py：根据带有预测偏移量的锚框来预测边界框，对预测边界框的置信度进行排序，使用非极大值抑制来预测边界框

Predict.py：定义类别和标签的预测函数

BaseBlocks.py：定义下采样块，基于下采样块定义，base_net，整合所有块，以及定义前向传播，返回特征图，锚框，类别预测，box偏移预测

Net.py：定义TinySSD网络

Train.py：定义损失函数和评价函数，训练模型，保存模型参数

Test.py：将边框格式转为matplotlib格式，显示边界框，加载模型参数进行预测

My_result.PNG：效果增强后加载'net_50_mine.pkl'训练出的结果

Original_result.PNG：效果增强之前加载'net_30_original.pkl'训练出的结果

net_30_origial.pkl：老师提供的权重文件

net_50_mine.pkl：数据增强后训练50轮保存的权重文件

四、注意事项

1、运行Test.py时，可能会有多个输出结果，单次只能显示一张，关闭后可继续显示测试结果。

2、detection文件夹需要和其余python文件在同一个目录下，否则需要重新改写文件路径。

3、老师的权重文件保存在net_30_original.pkl，数据增强后训练出的权重文件保存在net_50_mine.pkl，可以在Test.py中修改加载的权重文件。
