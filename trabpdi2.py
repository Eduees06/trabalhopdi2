import cv2
import numpy as np
from skimage import io, img_as_float
import matplotlib.pyplot as plt

def high_boost_filter(image, k, filter_size=5):
    # Converter a imagem para float
    image = img_as_float(image)
    
    # Aplicar filtro passa-baixa usando um filtro gaussiano
    low_pass = cv2.GaussianBlur(image, (filter_size, filter_size), 0)
    
    # High-boost
    high_boost = image + k * (image - low_pass)
    
    return np.clip(high_boost, 0, 1)

# Carregar a imagem
image_path = 'D:/TrabPDI/path_to_image.jpg'
image = io.imread(image_path, as_gray=True)

# Aplicar o filtro high-boost
k_values = [1, 1.5, 2, 2.5]
filtered_images = [high_boost_filter(image, k) for k in k_values]

# Exibir as imagens
fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

for i, (k, img) in enumerate(zip(k_values, filtered_images)):
    axes[i + 1].imshow(img, cmap='gray')
    axes[i + 1].set_title(f'k = {k}')
    axes[i + 1].axis('off')

plt.show()
