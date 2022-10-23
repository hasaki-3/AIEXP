# AIEXP🏫

一、运行方法

1、运行Train.py进行训练，自定轮数保存模型。

2、运行Test.py并加载训练好的模型进行测试，建议训练轮数大于30轮。

二、效果提升

使用transform扩充数据集进行效果提升，在Test.py中测试效果提升，置信度从0.83提升至0.95。

三、文件拆分

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

四、注意事项

1、运行Test.py时，可能会有多个输出结果，单次只能显示一张，关闭后可继续数据测试结果。

2、detection文件夹需要和python文件在同一个目录下。

3、老师的模型保存在net_30_teacher.pkl，我训练出的模型保存在net_30_mine.pkl，可以在Test.py中修改加载的模型。
