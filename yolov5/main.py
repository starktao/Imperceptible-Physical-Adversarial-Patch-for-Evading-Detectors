from patch_trainer import PatchTrainer
import glob
import os

def main():
    label_folder = '1112date/labels'  # 标签文件夹路径
    image_width = 640
    image_height = 640

    trainer = PatchTrainer(mode='custom_mode', label_folder=label_folder, 
                          image_width=image_width, image_height=image_height)

    folder_path = '1112date/images'  # 图像文件夹路径
    image_paths = glob.glob(os.path.join(folder_path, '*.png')) + \
                 glob.glob(os.path.join(folder_path, '*.jpg'))
    print("Image Paths:", image_paths)

    img_tensors = [trainer.preprocess_image(path) for path in image_paths]
    trainer.train(img_tensors)

if __name__ == '__main__':
    main() 