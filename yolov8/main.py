import os
import glob
from patch_trainer import PatchTrainer

def main():
    # 配置参数
    config = {
        'label_folder': '1127t2/labels',
        'image_width': 640,
        'image_height': 640,
        'image_folder': '1127t2/images'
    }

    # 初始化训练器
    trainer = PatchTrainer(mode='custom_mode', 
                          label_folder=config['label_folder'], 
                          image_width=config['image_width'], 
                          image_height=config['image_height'])

    # 获取所有图像路径
    image_paths = glob.glob(os.path.join(config['image_folder'], '*.png')) + \
                 glob.glob(os.path.join(config['image_folder'], '*.jpg'))
    print("Image Paths:", image_paths)

    # 预处理图像并开始训练
    img_tensors = [trainer.preprocess_image(path) for path in image_paths]
    trainer.train(img_tensors)

if __name__ == '__main__':
    main() 