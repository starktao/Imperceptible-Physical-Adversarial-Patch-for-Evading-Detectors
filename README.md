# 生成具有视觉隐藏性的对抗检测补丁

## 1.项目概述
研究表明，由深度神经网络生成的对抗性补丁能有效干扰计算机视觉模型，但这些补丁对人眼来说非常明显，并不适用于现实世界中的物理攻击。针对这一问题，我们提出了一种算法，用于生成具有深度视觉隐身效果的逃避检测补丁，以用于物理攻击。通过增强补丁与图像背景之间的相似性，我们提升了其隐身性，同时在单张图像上进行预训练以及在多张图像上进行联合训练，提高了其在不同距离下的攻击效果。一个 “多角度变换” 模块增强了其对不同角度检测的鲁棒性。我们的方法在攻击效果和隐身性方面均优于现有方法，能够同时躲避检测算法和人类视觉。
## 2.项目贡献
- 颜色相似度损失函数：通过提取补丁和应用背景的颜色相似度，生成相似度损失函数，将其加入到训练中去，从而提高了补丁的视觉隐藏性。
- 多角度变化模块：通过矩阵变化，模拟出现实场景下不同视角的补丁形态，对补丁的训练进行了增强，从而提高了补丁在不同拍摄角度下的攻击鲁棒性。
- 多图像联合训练：通过多张图像联合训练，代替单张图像轮次训练的传统训练方式，提高了补丁的攻击性。
- 项目针对yolov3，yolov5，甚至先进的yolov8都生成了对应的对抗补丁，据我所致，本项目是第一个生成攻击yolov8补丁的项目。
- 项目生成了用于攻击stop sign和车辆的隐藏对抗补丁，与以往的项目进行了对比。
## 3.项目部分展示
1. 与其他项目的对比

||我的|Hu等人的|Thys等人的|
|-|-|-|-|
|补丁|![cealpatch1](https://github.com/user-attachments/assets/4cd3e419-6b95-4129-931a-fa1a16dcf19b)|![cealpatch3](https://github.com/user-attachments/assets/e03650f1-7b0e-45ed-8e41-f3f13e847556)|![cealpatch5](https://github.com/user-attachments/assets/69935dd2-53c4-46d1-a31f-f3b066244d8d)|
|应用图|![cealres1](https://github.com/user-attachments/assets/422093e0-a3bc-4e1a-bbb1-29e85164f3d4)|![cealres3](https://github.com/user-attachments/assets/50dcb5a9-bbb4-4fa2-85f7-a8dca838b4f4)|![cealres5](https://github.com/user-attachments/assets/ee446328-cf26-4d52-8817-9666b57d3090)|
|项目链接||https://github.com/gordonjun2/Naturalistic-Adversarial-Patch|https://gitlab.com/EAVISE/adversarial-yolo|
2. 项目更多攻击stop sign和车辆的补丁应用前后的检测对比图

|攻击stop sign||||
|-|-|-|-|
|未应用补丁|![f9f84cb5a9389603376e86f183151f9](https://github.com/user-attachments/assets/919b8062-5955-4294-8ef1-4ada78f1c288)|![f26aee10caba52ae40b65d79de8eca2](https://github.com/user-attachments/assets/2231c1a7-b11c-4626-a79c-fc9c77d22d83)|![08816b43e9508531c28c1bda318da2a](https://github.com/user-attachments/assets/c8d4c7a8-36e1-4b5a-b0a0-b51af1777f30)|
|应用补丁后|![fe59cb26c0dc7401adbd10fbe173daf](https://github.com/user-attachments/assets/68c20b68-9f99-4e29-afcd-1d63518efb4b)|![d6638ed035f17826ec406c651a2a806](https://github.com/user-attachments/assets/f7deba57-3be3-401f-b733-0da0695c49c9)|![d080e45c18d9cb54f936a67e4911e6e](https://github.com/user-attachments/assets/abe25f80-f45a-40b9-a077-842bc5f68894)|

|攻击车辆||||
|-|-|-|-|
|未应用补丁|![frame6_120](https://github.com/user-attachments/assets/05138e5d-59f2-4c03-970b-1f23570154be)|![frame7_0](https://github.com/user-attachments/assets/48a1de47-3813-416a-9e5b-de83b7ae6e3c)|![frame10_85](https://github.com/user-attachments/assets/08a28627-284d-4b80-80b3-5a379359de70)|
|应用补丁后|![frame6_120_jpg rf bc445bcd316b17cd22ef29fa9905d53c](https://github.com/user-attachments/assets/81b8fa5f-dfe2-45b7-b177-a4c4f0f57ea5)|![frame7_0_jpg rf 138a75480c8b2ffc732ed33417c85b2b](https://github.com/user-attachments/assets/18ad0446-5a59-4dd9-930a-6c763da641bb)|![frame10_85_jpg rf 3a720f27ec8a422b6e68d998afd0e677](https://github.com/user-attachments/assets/d492c95e-548f-4c19-b96f-e8c66e34da88)|

# 注意
项目代码中分为v5和v8两个版本，在攻击yolov3和yolov5时，只需切换对应的models文件即可。相比yolov3和v5，yolov8删去了对象置信度的概念，而直接将类别置信度作为检测评判的标准，所以在提取图像张量的步骤有所不同，因此分为两个版本。
