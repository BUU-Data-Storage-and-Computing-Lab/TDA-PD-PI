import os
from skimage import io, color
import numpy as np
from ripser import Rips
from persim import PersistenceImager
from skimage.transform import resize

# 定义数据集路径
dataset_path = '/home/yanghan/桌面/finalcode/2024-DS/2024-Chinese Calligraphy Styles by Calligraphers/data/train/bdsr'

# 列出文件夹中的所有图像文件
image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]

# 创建 Rips 实例
rips = Rips()

# 处理每张图像
for image_file in image_files:
    try:
        # 构建完整的图像路径
        image_path = os.path.join(dataset_path, image_file)

        # 加载图像并将其转换为灰度
        img = io.imread(image_path)
        img_gray = color.rgb2gray(img)

        # H0 和 H1 的拓扑数据分析
        dgms = rips.fit_transform(img_gray.T)
        H0_dgm = dgms[0]
        H1_dgm = dgms[1]

        # 创建并绘制 H1 持久图
        pimgr_h1 = PersistenceImager(pixel_size=0.1)
        pimgr_h1.fit(H1_dgm)
        persistence_image_h1 = pimgr_h1.transform(H1_dgm)
        pimgr_h1.kernel_params = {'sigma': 0.5}

        # 将图像大小调整为224x224
        Final_PI_H1_224 = resize(persistence_image_h1, (224, 224), anti_aliasing=True)

        # 将浮点图像转换为8位整数格式
        Final_PI_H1_224_uint8 = (Final_PI_H1_224 * 255).astype(np.uint8)

        # 获取原始图像文件名（不包含扩展名）
        image_filename = os.path.splitext(os.path.basename(image_path))[0]

        # 构造保存路径，保持一致的文件名
        output_path = os.path.join('/home/yanghan/桌面/finalcode/2024-DS/1PI-CC/train/bdsr', f'{image_filename}.jpg')

        # 保存处理后的图片
        io.imsave(output_path, Final_PI_H1_224_uint8, quality=95)

    except Exception as e:
        print(f"处理图片时发生错误 {image_path}: {str(e)}")
