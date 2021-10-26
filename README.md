# cosineLossForImageClassfication

## 算法复现步骤：
1. 安装mindspore环境
2. 安装python环境
  cd arcface
  pip install -r requirements.txt
  pip install Augmentor # 数据增广库
3. 数据增广
  修改data_modify_augmentation.py文件中第52行训练数据路径，以及第53行数据保存路径。
  ![image](https://user-images.githubusercontent.com/36214675/138815551-7e363d5f-18ca-4d9d-ae55-134315cf3d2f.png)
  python data_modify_augmentation.py
  这个代码将每一类数据图片进行旋转、补边、透视仿射等数据增强，每一类规整到500张图片。本身超过500张图片的类别也被减少到500张，具体可以看运行出来的图片。
4. 训练（注意要修改训练数据路径，注释掉预训练模型路径代码,预训练是我断点继续训练才设置的）
  注释预训练模型部分代码：（train.py第198到203行；train_one_gpu.py第193到198行）
  ![image](https://user-images.githubusercontent.com/36214675/138815149-a2d8a514-1144-4103-8cba-0da4a9811382.png)
  多GPU：
  bash scripts/run_distribute_train.sh /data/ccl/RP2K_rp2k_dataset/all/train_small/ /home/ccl/Documents/codes/python/minds/models/research/cv/arcface/r.json
  单GPU：
  python train_one_gpu.py
5. 模型评测
  运行python inferece_all.py对每一类数据图片进行特征向量提取，每张图提取出512维的特征向量。求取每类的平均值，作为该类的中心向量。并将结果保存在json文件中。（注意在第52行修改json文件保存的路径；在41行修改模型路径）
  ![image](https://user-images.githubusercontent.com/36214675/138816410-f5df5b56-e2f2-4586-a96c-a3163f535afd.png)
  运行python inferece_all_check.py，读取json文件获得每个类别的中心向量，推理每张测试集的图片提取特征向量，并求向量与每类中心向量的余弦相似度，相似度最大的一类作为最终分类。打印预测正确的数量count_right和预测错误的数量count_wrong,准确率计算count_right/(count_right + count_wrong)=0.79527
  注：先对每类图像聚类出多个类中心会改善这个指标，后面有时间再写代码。
