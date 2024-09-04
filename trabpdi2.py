import numpy as np
from skimage import io, img_as_float
import cv2
import matplotlib.pyplot as plt

def kernel_gaussiano(tamanho, sigma=1):
    eixo = np.linspace(-(tamanho // 2), tamanho // 2, tamanho)
    kernel_1d = np.exp(-np.square(eixo) / (2 * np.square(sigma)))
    kernel_1d /= np.sum(kernel_1d)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum()

def aplicar_filtro(imagem, kernel):
    imagem_filtrada = cv2.filter2D(imagem, -1, kernel)
    return imagem_filtrada

def filtro_high_boost(imagem, k, tamanho_filtro=5, sigma=1):
    imagem = img_as_float(imagem)
    
    # Criar e aplicar o filtro gaussiano
    gauss_kernel = kernel_gaussiano(tamanho_filtro, sigma)
    passa_baixa = aplicar_filtro(imagem, gauss_kernel)
    
    # High-boost
    high_boost = imagem + k * (imagem - passa_baixa)
    
    return np.clip(high_boost, 0, 1)

# Carregar a imagem
caminho_imagem = 'D:/TrabPDI/path_to_image.jpg'
imagem = io.imread(caminho_imagem, as_gray=True)

# Aplicar o filtro high-boost
valores_k = [5, 30, 50, 100]
imagens_filtradas = [filtro_high_boost(imagem, k) for k in valores_k]

# Exibir as imagens
fig, eixos = plt.subplots(1, len(valores_k) + 1, figsize=(15, 5))
eixos[0].imshow(imagem, cmap='gray')
eixos[0].set_title('Original')
eixos[0].axis('off')

for i, (k, img) in enumerate(zip(valores_k, imagens_filtradas)):
    eixos[i + 1].imshow(img, cmap='gray')
    eixos[i + 1].set_title(f'k = {k}')
    eixos[i + 1].axis('off')

plt.show()