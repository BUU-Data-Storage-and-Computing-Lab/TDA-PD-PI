import os
from skimage import io, color
import numpy as np
from ripser import Rips
from persim import PersistenceImager
from skimage.transform import resize
from PIL import Image

# 定义数据集路径
dataset_path = '/home/yanghan/桌面/finalcode/2024-DS/2024-Chinese Calligraphy Styles by Calligraphers/data/val/yyr'

# 列出文件夹中的所有图像文件
image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]

# 创建 Rips 实例
rips_instance = Rips()

# 处理每张图像
for image_file in image_files:
    try:
        # 构建完整的图像路径
        image_path = os.path.join(dataset_path, image_file)

        # 加载图像并将其转换为灰度
        img = io.imread(image_path)
        img_gray = color.rgb2gray(img)

        # H0 的拓扑数据分析
        rips_h0 = Rips()
        dgms_h0 = rips_h0.fit_transform(img_gray.T)
        H0_dgm = dgms_h0[0]

        # 用最大有限值替换无穷大
        max_finite_value = np.max(H0_dgm[H0_dgm[:, 1] != np.inf, 1])
        H0_dgm[H0_dgm[:, 1] == np.inf, 1] = max_finite_value

        # 添加一个新的点到 H0_dgm
        new_point_h0 = np.array([[0.5, 0.6]])
        H0_dgm = np.vstack([H0_dgm, new_point_h0])

        # 创建并绘制 H0 持久图
        pimgr_h0 = PersistenceImager(pixel_size=0.1, kernel_params={'sigma': 0.5})
        pimgr_h0.fit(H0_dgm)
        persistence_image_h0 = pimgr_h0.transform(H0_dgm)

        # 将图像大小调整为224x224
        Final_PI_H0_224 = resize(persistence_image_h0, (224, 224), anti_aliasing=True)
        # 将浮点图像转换为8位整数格式
        Final_PI_H0_224_uint8 = (Final_PI_H0_224 * 255).astype(np.uint8)

        # 获取原始图像文件名（不包含扩展名）
        image_filename = os.path.splitext(os.path.basename(image_path))[0]

        # 构造保存路径，保持一致的文件名
        output_path = os.path.join('/home/yanghan/桌面/finalcode/2024-DS/0PI-CC/val/yyr', f'{image_filename}.jpg')

        # 保存处理后的图片
        img_pil = Image.fromarray(Final_PI_H0_224_uint8)
        img_pil.save(output_path, quality=95)

    except Exception as e:
        print(f"处理图片时发生错误 {image_path}: {str(e)}")
