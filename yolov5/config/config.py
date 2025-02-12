class Config:
    # 训练相关配置
    BATCH_SIZE = 7
    EPOCHS_PER_BATCH = 1000
    LEARNING_RATE = 0.1
    SCHEDULER_STEP_SIZE = 500
    SCHEDULER_GAMMA = 0.1
    
    # 路径配置
    SAVE_IMAGE_DIR = "saved_img530"
    SAVE_PATCH_DIR = "saved_patches5310"
    
    # 模型配置
    MODEL_WEIGHTS = "yolov5s.pt"
    
    # 其他参数
    TARGET_CLASS = 2
    IMAGE_SIZE = 640 