# AIEXP

可以运行Train.py进行训练，自定轮数保存模型。

运行Test.py并加载训练好的模型进行测试，建议训练轮数大于30轮。

使用transform扩充数据集进行效果提升，在Test.py中测试效果提升，置信度从0.83提升至0.95。

read_data.py：数据读取，进行数据增强，数据加载并转为pytorch可用的数据

GenAnchors.py：针对每个像素点生成锚框，确定偏移量和中心点

gen_gtboxes.py：将真实边界框分配给锚框->计算锚框与真实边界框之间的偏移量->使用真实边界框标记每个锚框 ：如果⼀个锚框没有被分配，，标记其为背景，mask = [0, 0, 0, 0]

iou.py：计算锚框与锚框，锚框与gtbox之间的iou

loss.py：计算标签损失(交叉熵损失函数)和类别损失(L1损失函数)

nms.py：找出含有物体的锚框，采用置信度和非极大值抑制置信度筛选出最终需要的锚框(预测阶段使用)

prediction.py：定义类别和标签的预测函数，由于存在多个预测输出，需要先将预测结果压平，再连接起来

网络模块：

blocks.py：定义下采样块，基础网络块，以及后续块的整合方法->blocks.py

图片：

catdog.jpg：锚框绘制

banana.png：结果预测

需要关闭图像窗口才可以继续训练
