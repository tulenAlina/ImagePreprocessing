import os
import json
import numpy as np
import cv2
import pydicom
import logging
from skimage import exposure, filters
from skimage.morphology import erosion, dilation, square, disk
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation
from skimage.metrics import structural_similarity as ssim
from math import exp
from skimage.util import random_noise
from flask import Flask, request, jsonify
from cryptography.fernet import Fernet
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
import io
import random
import skimage.util
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import urllib.request
import tarfile
import zipfile
import shutil
from tqdm import tqdm
import time
import sys

PROCESSED_DATA_FILE = 'processed_data.npz'

# Установка кодировки вывода
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)

# Модуль ввода/вывода данных
class ImageIO:
    def __init__(self):
        self.supported_formats = ['.dcm', '.png', '.jpg', '.jpeg', '.bmp']
        
    def load_image(self, file_path):
        """Загрузка изображения с автоматическим определением формата"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.dcm':
            try:
                ds = pydicom.dcmread(file_path)
                image = ds.pixel_array
                metadata = {
                    'patient_id': ds.get('PatientID', ''),
                    'study_date': ds.get('StudyDate', ''),
                    'modality': ds.get('Modality', '')
                }
                return image, metadata
            except Exception as e:
                raise ValueError(f"Error reading DICOM file: {str(e)}")
        else:
            try:
                # Try with OpenCV first
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    # Fallback to PIL if OpenCV fails
                    pil_img = Image.open(file_path)
                    image = np.array(pil_img)
                    if len(image.shape) == 3 and image.shape[2] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                return image, {}
            except Exception as e:
                raise ValueError(f"Error loading image: {str(e)}")

    def normalize_image(self, image):
        """Нормализация динамического диапазона"""
        if image is None:
            raise ValueError("Image is None")
            
        if image.dtype == np.uint16:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        elif image.dtype == np.float32 or image.dtype == np.float64:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return image

    def convert_color_space(self, image, target_space='GRAY'):
        """Преобразование цветовых пространств"""
        if image is None:
            raise ValueError("Image is None")
            
        if len(image.shape) == 3 and image.shape[2] == 3 and target_space == 'GRAY':
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2 and target_space == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

# Модуль классической обработки
class ClassicalProcessing:
    def gaussian_filter(self, image, sigma=1.0):
        """Гауссово размытие"""
        if image is None:
            raise ValueError("Image is None")
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)

    def median_filter(self, image, size=3):
        """Медианный фильтр"""
        if image is None:
            raise ValueError("Image is None")
        return cv2.medianBlur(image, size)

    def bilateral_filter(self, image, d=9, sigma_color=75, sigma_space=75):
        """Билатеральный фильтр с улучшенной обработкой ошибок"""
        try:
            if image is None:
                raise ValueError("Изображение не загружено (None)")
            
            # Конвертируем в grayscale если нужно
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Проверка типа данных
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # Проверка параметров
            if d <= 0 or sigma_color <= 0 or sigma_space <= 0:
                raise ValueError(
                    f"Некорректные параметры: d={d}, sigma_color={sigma_color}, sigma_space={sigma_space}"
                )
            
            return cv2.bilateralFilter(image, d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
        except Exception as e:
            error_msg = f"Ошибка bilateral фильтра: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
            return image  # Возвращаем оригинал при ошибке

    def clahe(self, image, clip_limit=2.0, grid_size=(8,8)):
        """Адаптивная эквализация гистограммы"""
        if image is None:
            raise ValueError("Image is None")
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(image)

    def morphological_operation(self, image, operation='dilate', size=3, iterations=1):
        """Морфологические операции"""
        if image is None:
            raise ValueError("Image is None")
            
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        if operation == 'erode':
            return cv2.erode(image, kernel, iterations=iterations)
        elif operation == 'dilate':
            return cv2.dilate(image, kernel, iterations=iterations)
        elif operation == 'open':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'gradient':
            return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        return image

    def sharpen(self, image, strength=1.0, kernel_size=3):
        """Увеличение резкости изображения с настраиваемыми параметрами
        
        Args:
            image: входное изображение
            strength: сила эффекта резкости (по умолчанию 1.0)
            kernel_size: размер ядра (3 или 5, по умолчанию 3)
        
        Returns:
            Увеличенное по резкости изображение
        """
        if image is None:
            raise ValueError("Image is None")
        
        # Проверка допустимых размеров ядра
        if kernel_size not in [3, 5]:
            raise ValueError("kernel_size must be 3 or 5")
        
        # Конвертируем в float32 для точных вычислений
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # Генерация ядра в зависимости от размера
        if kernel_size == 3:
            kernel = np.array([
                [0,          -0.3*strength, 0],
                [-0.3*strength, 1 + 0.8*strength, -0.3*strength],
                [0,          -0.3*strength, 0]
            ])
        else:  # kernel_size == 5
            kernel = np.array([
                [0,  0,        -0.1*strength, 0,        0],
                [0, -0.2*strength, -0.3*strength, -0.2*strength, 0],
                [-0.1*strength, -0.3*strength, 1 + 1.2*strength, -0.3*strength, -0.1*strength],
                [0, -0.2*strength, -0.3*strength, -0.2*strength, 0],
                [0,  0,        -0.1*strength, 0,        0]
            ])
        
        # Нормализация ядра (сумма = 1)
        kernel = kernel / (1.0 + 0.2*strength)
        
        # Применяем фильтр
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Обрезаем значения и возвращаем в [0,255]
        sharpened = (sharpened - sharpened.min()) * (255.0 / (sharpened.max() - sharpened.min()))
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # Отладочная информация
        print(f"Sharpening - kernel: {kernel_size}x{kernel_size}, strength: {strength}, "
            f"input range: [{image.min():.2f}, {image.max():.2f}], "
            f"output range: [{sharpened.min()}, {sharpened.max()}]")
        
        return sharpened

    def edge_detection(self, image, method='canny', low_threshold=50, high_threshold=150):
        """Обнаружение границ"""
        if image is None:
            raise ValueError("Image is None")
            
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        if method == 'canny':
            return cv2.Canny(image, low_threshold, high_threshold)
        elif method == 'sobel':
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
            return cv2.magnitude(sobelx, sobely)
        return image

    def threshold(self, image, thresh=127, maxval=255, type='binary'):
        """Пороговая обработка"""
        if image is None:
            raise ValueError("Image is None")
            
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        if type == 'binary':
            _, result = cv2.threshold(image, thresh, maxval, cv2.THRESH_BINARY)
        elif type == 'otsu':
            _, result = cv2.threshold(image, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif type == 'adaptive':
            result = cv2.adaptiveThreshold(image, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        return result

# Модуль нейросетевой обработки
class DenoisingAutoencoder(nn.Module):
    """Автоэнкодер для шумоподавления"""
    def __init__(self):
        super().__init__()
        # Энкодер с batch-norm
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),  # 224 -> 112
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),  # 112 -> 56
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Декодер с skip-connections
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),  # 56 -> 112
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),  # 112 -> 224
            
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh()
        )
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Энкодер с padding=1 для сохранения размеров
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Центральная часть
        self.center = self.conv_block(512, 1024)
        
        # Декодер с корректными параметрами
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
        # MaxPool с чётким уменьшением размеров
        self.pool = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling should be defined in __init__
        self.adaptive_pool = nn.AdaptiveAvgPool2d((224, 224))

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Сохраняем оригинальный размер
        original_size = x.size()[2:]
        
        # Приводим к фиксированному размеру 224x224
        x = self.adaptive_pool(x)
        
        # Энкодер
        enc1 = self.enc1(x)  # 224->224
        enc2 = self.enc2(self.pool(enc1))  # 224->112->112
        enc3 = self.enc3(self.pool(enc2))  # 112->56->56
        enc4 = self.enc4(self.pool(enc3))  # 56->28->28
        
        # Центральный блок
        center = self.center(self.pool(enc4))  # 28->14->14
        
        # Декодер
        dec4 = self.up4(center)  # 14->28
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)  # 28->28
        
        dec3 = self.up3(dec4)  # 28->56
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)  # 56->56
        
        dec2 = self.up2(dec3)  # 56->112
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)  # 112->112
        
        dec1 = self.up1(dec2)  # 112->224
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)  # 224->224
        
        # Финальный слой
        output = self.final(dec1)
        
        # Возвращаем к исходному размеру
        output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        
        return torch.sigmoid(output)
    
class NeuralProcessing:
    def __init__(self, device='cuda'):
        # Автоматически определяем доступное устройство
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.ssim = SSIM(window_size=11, size_average=True)
        self.denoiser = DenoisingAutoencoder().to(self.device)
        self.unet = UNet().to(self.device)
        self.criterion = nn.MSELoss()
        self.denoiser_optimizer = optim.Adam(self.denoiser.parameters(), lr=0.001)
        self.unet_optimizer = optim.Adam(self.unet.parameters(), lr=0.0001)
        self.ssim = SSIM(window_size=11, size_average=True)
        self.denoiser_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        self.denoiser_optimizer, mode='min', patience=3, factor=0.5)
        
    def train_denoiser(self, dataloader, epochs=10):
        self.denoiser.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            for noisy, clean in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                self.denoiser_optimizer.zero_grad()
                outputs = self.denoiser(noisy)
                
                # Добавляем perceptual loss
                mse_loss = self.criterion(outputs, clean)
                
                # Добавляем SSIM loss для сохранения структур
                ssim_loss = 1 - torch.mean(torch.clamp(
                    (1 + self.ssim(outputs, clean, window_size=11)) / 2, 0, 1))
                
                # Комбинированный лосс
                loss = mse_loss + 0.5 * ssim_loss
                
                loss.backward()
                self.denoiser_optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            
            # Сохраняем лучшую модель
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.denoiser.state_dict(), 'weights/best_denoiser.pth')
        
        self.denoiser_scheduler.step(avg_loss)
        # Загружаем лучшие веса
        self.denoiser.load_state_dict(torch.load('weights/best_denoiser.pth'))
    
    def train_unet(self, dataloader, epochs=10):
        self.unet.train()
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.unet_optimizer.zero_grad()
                outputs = self.unet(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.unet_optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    def load_pretrained_weights(self):
        """Инициализация весов с возможностью загрузки"""
        # Пути к файлам весов
        denoiser_path = 'weights/denoiser.pth'
        unet_path = 'weights/unet.pth'
        
        # Для DenoisingAutoencoder
        if os.path.exists(denoiser_path):
            self.denoiser.load_state_dict(torch.load(denoiser_path))
        else:
            nn.init.xavier_uniform_(self.denoiser.encoder[0].weight)
            nn.init.xavier_uniform_(self.denoiser.decoder[0].weight)
        
        # Для UNet
        if os.path.exists(unet_path):
            self.unet.load_state_dict(torch.load(unet_path))
        else:
            for m in self.unet.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def denoise_image(self, image):
        # Конвертация в grayscale если нужно
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Сохраняем оригинальные параметры
        orig_min, orig_max = image.min(), image.max()
        orig_dtype = image.dtype
        
        # Нормализация в [0, 1] с сохранением динамического диапазона
        if orig_max > 1:  # Если изображение не нормализовано
            image_norm = image.astype(np.float32) / 255.0
        else:
            image_norm = image.astype(np.float32)
        
        # Добавляем batch и channel размерности
        tensor = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.denoiser(tensor)
        
        # Конвертируем обратно в numpy
        output = output.squeeze().cpu().numpy()
        
        # Восстанавливаем динамический диапазон
        output = (output + 1) * 127.5  # Конвертация из [-1, 1] в [0, 255]
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        # Мягкая постобработка
        output = cv2.medianBlur(output, 3)
        
        # Адаптивная гистограммная коррекция (более мягкая)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
        output = clahe.apply(output)
        
        return output

    def enhance_details(self, image):
        if image is None:
            raise ValueError("Image is None")
            
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        original_dtype = image.dtype
        # Более мягкая нормализация
        image = image.astype(np.float32) / 255.0
        
        # Конвертация в тензор
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        # Добавляем паддинг если нужно
        if tensor.size(2) % 32 != 0 or tensor.size(3) % 32 != 0:
            h, w = tensor.size(2), tensor.size(3)
            new_h = ((h // 32) + 1) * 32
            new_w = ((w // 32) + 1) * 32
            tensor = F.pad(tensor, (0, new_w - w, 0, new_h - h), mode='reflect')
        
        with torch.no_grad():
            output = self.unet(tensor)
        
        # Обрезаем паддинг если добавляли
        if tensor.size(2) != output.size(2):
            output = output[:, :, :h, :w]
        
        output = output.squeeze().cpu().numpy()
        
        # Мягкая постобработка
        output = (output * 255).astype(np.uint8)
        output = cv2.medianBlur(output, 3)  # Легкое сглаживание
        
        return output

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def create_window(self, window_size, channel):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2 / float(2 * 1.5**2)) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        window = torch.outer(gauss, gauss)
        return window.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1, img2):
        window = self.window.to(img1.device)
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class DenoisingDataset(Dataset):
    def __init__(self, clean_images, noise_level=15, target_size=(224, 224)):  # Уменьшили noise_level
        self.clean_images = []
        for img in clean_images:
            # Нормализация и сохранение
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_resized = cv2.resize(img_norm, target_size)
            self.clean_images.append(img_resized)
        
        self.noise_level = noise_level
        self.target_size = target_size
        
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        self.denoiser.apply(init_weights)

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean = (clean.astype(np.float32) / 127.5) - 1.0  
        noisy = (noisy.astype(np.float32) / 127.5) - 1.0
        
        # Случайное отражение/поворот
        if random.random() > 0.5:
            clean = cv2.flip(clean, 1)
        if random.random() > 0.5:
            clean = cv2.rotate(clean, cv2.ROTATE_90_CLOCKWISE)
        
        # Реалистичный шум
        noise_type = random.choice(['gaussian', 'poisson', 'speckle'])
        if noise_type == 'gaussian':
            noisy = random_noise(clean, mode='gaussian', var=(self.noise_level/255)**2)
        elif noise_type == 'poisson':
            noisy = random_noise(clean, mode='poisson')
        else:
            noisy = random_noise(clean, mode='speckle')
        
        noisy = (noisy * 255).astype(np.uint8)
        
        # Нормализация
        clean = (clean.astype(np.float32) / 127.5) - 1.0
        noisy = (noisy.astype(np.float32) / 127.5) - 1.0
        
        return (
            torch.from_numpy(noisy).unsqueeze(0).float(),
            torch.from_numpy(clean).unsqueeze(0).float()
        )

# Гибридный обработчик
class HybridProcessor:
    def __init__(self):
        self.io = ImageIO()
        self.classical = ClassicalProcessing()
        self.neural = NeuralProcessing()
        self.config = self.load_config('config.json')

    def load_config(self, config_path):
        """Загрузка конфигурации обработки"""
        default_config = {
            "pipeline": [
                {
                    "name": "clahe_correction",
                    "type": "classical",
                    "method": "clahe",
                    "params": {
                        "clip_limit": 2.0,
                        "grid_size": [8, 8]
                    }
                },
                {
                    "name": "bilateral_filter",
                    "type": "classical",
                    "method": "bilateral",
                    "params": {
                        "d": 9,
                        "sigma_color": 75,
                        "sigma_space": 75
                    }
                },
                {
                    "name": "denoising_autoencoder",
                    "type": "neural",
                    "model": "denoiser"
                },
                {
                    "name": "unet_enhancement",
                    "type": "neural",
                    "model": "unet"
                },
                {
                    "name": "sharpening",
                    "type": "classical",
                    "method": "sharpen"
                }
            ]
        }
        
        try:
            with open(config_path) as f:
                return json.load(f)
        except:
            print(f"Warning: Could not load config file {config_path}, using default configuration")
            return default_config

    def process_pipeline(self, image_path):
        try:
            image, metadata = self.io.load_image(image_path)
            if image is None:
                raise ValueError("Loaded image is None")
                
            # Сохраняем оригинальный размер и тип
            original_size = image.shape[:2]
            original_dtype = image.dtype
            
            # Приводим к стандартному размеру для обработки
            image = cv2.resize(image, (224, 224))
            image = self.io.normalize_image(image)
            
            # Обработка
            results = {'original': cv2.resize(image, (original_size[1], original_size[0]))}
            
            for step in self.config['pipeline']:
                try:
                    if step['type'] == 'classical':
                        image = self.apply_classical(image, step)
                    elif step['type'] == 'neural':
                        # Для нейросетевой обработки сохраняем промежуточный размер
                        processed = self.apply_neural(image, step)
                        image = cv2.resize(processed, (224, 224))  # Возвращаем к размеру для следующей обработки
                    
                    # Все результаты сохраняем в исходном размере
                    resized_result = cv2.resize(image, (original_size[1], original_size[0]))
                    # Восстанавливаем исходный тип данных
                    if original_dtype != resized_result.dtype:
                        resized_result = resized_result.astype(original_dtype)
                    results[step['name']] = resized_result.copy()
                    
                except Exception as e:
                    print(f"Error in {step['name']}: {str(e)}")
                    continue
            
            return results, metadata
        except Exception as e:
            print(f"Error in process_pipeline: {str(e)}")
            return {}, {}

    def apply_classical(self, image, params):
        """Применение классических методов с полной обработкой ошибок"""
        try:
            if not isinstance(image, np.ndarray):
                raise TypeError("Изображение должно быть numpy массивом")
            
            if image.size == 0:
                raise ValueError("Пустое изображение")
                
            # Проверка минимального размера
            if min(image.shape[:2]) < 10:
                raise ValueError("Слишком маленькое изображение")
                
            method = str(params.get('method', '')).lower()
            if not method:
                raise ValueError("Не указан метод обработки")
                
            if image is None or not isinstance(image, np.ndarray):
                raise ValueError("Некорректное входное изображение")
                
            # Основная логика обработки
            processor = getattr(self.classical, method, None)
            if not processor:
                raise AttributeError(f"Метод {method} не найден")
                
            # Применение параметров
            method_params = params.get('params', {})
            if not isinstance(method_params, dict):
                method_params = {}
                
            result = processor(image, **method_params)
        
            # Нормализуем результат
            if result.dtype != np.uint8:
                result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
            return result
            
        except Exception as e:
            error_msg = f"Ошибка в {params.get('name', 'unknown')}: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
            return image

    def apply_neural(self, image, params):
        """Применение нейросетевых методов"""
        model = params['model']
        
        if model == 'denoiser':
            return self.neural.denoise_image(image)
        elif model == 'unet':
            return self.neural.enhance_details(image)
        else:
            raise ValueError(f"Unknown neural model: {model}")

# Модуль оценки качества
class QualityAssessment:
    def calculate_psnr(self, original, processed):
        """Вычисление PSNR"""
        if original is None or processed is None:
            raise ValueError("Images cannot be None")
            
        h = min(original.shape[0], processed.shape[0])
        w = min(original.shape[1], processed.shape[1])
        
        original = original[:h, :w]
        processed = processed[:h, :w]
    
        mse = np.mean((original - processed) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))

    def calculate_ssim(self, original, processed):
        """Вычисление SSIM"""
        if original is None or processed is None:
            raise ValueError("Images cannot be None")
            
        from skimage.metrics import structural_similarity as ssim
        
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
        h = min(original.shape[0], processed.shape[0])
        w = min(original.shape[1], processed.shape[1])
        
        original = original[:h, :w]
        processed = processed[:h, :w]
        
        return ssim(original, processed, data_range=255)

    def generate_report(self, results, output_path):
        """Генерация отчета"""
        if not results:
            raise ValueError("No results to generate report")
            
        os.makedirs(output_path, exist_ok=True)
        
        fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
        if len(results) == 1:
            axes = [axes]
            
        for ax, (name, img) in zip(axes, results.items()):
            if img is None:
                raise ValueError(f"Image {name} is None")
                
            if len(img.shape) == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(name)
            ax.axis('off')
        
        report_path = os.path.join(output_path, 'report.png')
        plt.tight_layout()
        plt.savefig(report_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Сохранение метрик
        metrics = {}
        steps = list(results.keys())
        for i in range(1, len(steps)):
            metrics[f'{steps[i-1]}_to_{steps[i]}'] = {
                'psnr': self.calculate_psnr(results[steps[i-1]], results[steps[i]]),
                'ssim': self.calculate_ssim(results[steps[i-1]], results[steps[i]])
            }
        
        with open(os.path.join(output_path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return report_path

# REST API интерфейс
@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Создание временных папок
    os.makedirs('temp', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    temp_path = os.path.join('temp', file.filename)
    
    try:
        # Сохранение временного файла
        file.save(temp_path)
        
        # Проверка, что файл сохранен
        if not os.path.exists(temp_path):
            return jsonify({'error': 'Failed to save temporary file'}), 500
            
        # Обработка изображения
        start_time = time.time()
        processor = HybridProcessor()
        results, metadata = processor.process_pipeline(temp_path)
        
        # Оценка качества
        qa = QualityAssessment()
        
        # Генерация отчета
        report_dir = os.path.join('reports', os.path.splitext(file.filename)[0])
        report_path = qa.generate_report(results, report_dir)
        
        # Чтение метрик
        with open(os.path.join(report_dir, 'metrics.json')) as f:
            metrics = json.load(f)
        
        # Возврат результата
        response = {
            'status': 'success',
            'metadata': metadata,
            'metrics': metrics,
            'report': report_path,
            'processing_time': round(time.time() - start_time, 2)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        
    finally:
        # Очистка временных файлов
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Шифрование данных
class DataSecurity:
    def __init__(self, key_path='encryption.key'):
        self.key = self.load_or_generate_key(key_path)
        self.cipher = Fernet(self.key)

    def load_or_generate_key(self, path):
        """Загрузка или генерация ключа шифрования"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(path, 'wb') as f:
                f.write(key)
            return key

    def encrypt_data(self, data):
        """Шифрование данных"""
        if isinstance(data, str):
            data = data.encode()
        return self.cipher.encrypt(data)

    def decrypt_data(self, encrypted_data):
        """Дешифрование данных"""
        return self.cipher.decrypt(encrypted_data).decode()

# Функции для добавления артефактов
def add_gaussian_noise(image, sigma=10):
    """Добавление гауссова шума"""
    if image is None:
        raise ValueError("Image is None")
        
    if image.dtype == np.uint8:
        noise = np.random.normal(0, sigma, image.shape).astype(np.uint8)
        return cv2.add(image, noise)
    else:
        noise = np.random.normal(0, sigma, image.shape).astype(image.dtype)
        return np.clip(image + noise, 0, 255).astype(image.dtype)

def add_impulse_noise(image, amount=0.05):
    """Добавление импульсного шума"""
    if image is None:
        raise ValueError("Image is None")
        
    noisy_image = skimage.util.random_noise(image, mode='s&p', amount=amount)
    return (noisy_image * 255).astype(np.uint8)

def adjust_contrast(image, min_hu=30, max_hu=170):
    """Коррекция контраста"""
    if image is None:
        raise ValueError("Image is None")
        
    # Сжатие динамического диапазона
    image = np.clip(image, min_hu, max_hu)
    # Нормализация к диапазону [0, 255]
    image = ((image - min_hu) / (max_hu - min_hu)) * 255
    return image.astype(np.uint8)

def add_uneven_illumination(image, sigma=200):
    """Добавление неравномерного освещения"""
    if image is None:
        raise ValueError("Image is None")
        
    rows, cols = image.shape[:2]
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    center_x, center_y = cols // 2, rows // 2
    gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    gaussian = (gaussian / np.max(gaussian)) * 50  # Масштабирование
    
    # Применение пятна к изображению
    if len(image.shape) == 3:
        gaussian = np.expand_dims(gaussian, axis=-1)
    return np.clip(image + gaussian, 0, 255).astype(np.uint8)

def apply_artifacts(image):
    """Применение всех артефактов"""
    if image is None:
        raise ValueError("Image is None")
        
    image = add_gaussian_noise(image)
    image = add_impulse_noise(image)
    image = adjust_contrast(image)
    image = add_uneven_illumination(image)
    return image

# Функция для загрузки и подготовки изображений
def load_and_prepare_data(base_dir, target_size=(224, 224), test_size=0.2):
    """Загрузка данных с учётом вложенной структуры"""
    # Проверяем существование сохраненных данных
    if os.path.exists(PROCESSED_DATA_FILE):
        print("Загружаем предварительно обработанные данные...")
        data = np.load(PROCESSED_DATA_FILE)
        return data['X_train'], data['X_test'], data['y_train'], data['y_test']
    
    images = []
    labels = []
    
    train_dir = os.path.join(base_dir, 'chest_xray', 'train')
    test_dir = os.path.join(base_dir, 'chest_xray', 'test')
    
    print(f"Ищем данные в: {train_dir}")
    print(f"Ищем данные в: {test_dir}")

    for class_name in ['NORMAL', 'PNEUMONIA']:
        # Обработка тренировочных данных
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Предупреждение: директория {class_dir} не найдена")
            continue
            
        for filename in tqdm(os.listdir(class_dir), desc=f'Loading {class_name} train'):
            if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                img_path = os.path.join(class_dir, filename)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, target_size)
                        if img.dtype != np.uint8:
                            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        images.append(img)
                        labels.append(0 if class_name == 'NORMAL' else 1)
                except Exception as e:
                    print(f"Ошибка загрузки {img_path}: {str(e)}")
        
        # Обработка тестовых данных
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Предупреждение: директория {class_dir} не найдена")
            continue
            
        for filename in tqdm(os.listdir(class_dir), desc=f'Loading {class_name} test'):
            if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                img_path = os.path.join(class_dir, filename)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, target_size)
                        if img.dtype != np.uint8:
                            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        images.append(img)
                        labels.append(0 if class_name == 'NORMAL' else 1)
                except Exception as e:
                    print(f"Ошибка загрузки {img_path}: {str(e)}")

    if not images:
        raise ValueError("Не найдено ни одного изображения в указанных директориях")

    images = np.array(images)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, 
        test_size=test_size, 
        random_state=42,
        stratify=labels
    )
    
    # Сохраняем данные для последующих запусков
    np.savez(PROCESSED_DATA_FILE, 
             X_train=X_train, X_test=X_test, 
             y_train=y_train, y_test=y_test)
    
    return X_train, X_test, y_train, y_test

def validate_dataset_structure(base_dir):
    """Проверка структуры с учётом вложенности"""
    required_paths = {
        'train': os.path.join(base_dir, 'chest_xray', 'train'),
        'test': os.path.join(base_dir, 'chest_xray', 'test')
    }
    
    for split, path in required_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Основная директория не найдена: {path}")
        
        for cls in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(path, cls)
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"Директория класса не найдена: {class_dir}")
                
            files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            if not files:
                raise ValueError(f"Директория {class_dir} не содержит изображений")
            print(f"Найдено {len(files)} изображений в {class_dir}")

def create_test_dataset(dataset_dir):
    """Создание тестового датасета"""
    print("Creating test dataset...")
    for dataset_type in ['train', 'test']:
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(dataset_dir, dataset_type, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(5):  # 5 изображений каждого класса
                # Создаем 8-битные изображения сразу
                if class_name == 'NORMAL':
                    img = np.random.randint(100, 200, (224, 224), dtype=np.uint8)
                else:
                    img = np.random.randint(0, 100, (224, 224), dtype=np.uint8)
                    img[100:150, 100:150] = np.random.randint(150, 255, (50, 50), dtype=np.uint8)
                
                # Сохраняем как JPEG
                cv2.imwrite(os.path.join(class_dir, f"{class_name.lower()}_{i}.jpeg"), img, 
                           [int(cv2.IMWRITE_JPEG_QUALITY), 90])

# Функция для применения артефактов к набору изображений
def apply_artifacts_to_images(images):
    """Применение артефактов к набору изображений"""
    artifact_images = []
    for img in images:
        artifact_img = apply_artifacts(img)
        artifact_images.append(artifact_img)
    return np.array(artifact_images)

# Функция для нормализации изображений
def normalize_images(images):
    """Нормализация изображений"""
    normalized_images = []
    for img in images:
        normalized_img = img.astype('float32') / 255.0
        normalized_images.append(normalized_img)
    return np.array(normalized_images)

# Функция для создания структуры папок
def create_data_structure(base_dir):
    """Создание структуры папок для данных"""
    os.makedirs(os.path.join(base_dir, 'train', 'normal'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'train', 'pneumonia'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'validation', 'normal'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'validation', 'pneumonia'), exist_ok=True)

# Функция для сохранения изображений в соответствующие папки
def save_images(images, labels, base_dir, dataset_type):
    """Сохранение изображений с разметкой"""
    for i, (image, label) in enumerate(zip(images, labels)):
        if label == 0:  # normal
            file_path = os.path.join(base_dir, dataset_type, 'normal', f'normal_{i}.png')
        else:  # pneumonia
            file_path = os.path.join(base_dir, dataset_type, 'pneumonia', f'pneumonia_{i}.png')
        cv2.imwrite(file_path, image)

# Функция для скачивания и распаковки датасета
def download_and_extract_dataset():
    """Скачивание и распаковка датасета"""
    dataset_dir = "chest_xray"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Проверяем наличие уже распакованных данных
    required_dirs = [
        os.path.join(dataset_dir, 'train', 'NORMAL'),
        os.path.join(dataset_dir, 'train', 'PNEUMONIA'),
        os.path.join(dataset_dir, 'test', 'NORMAL'), 
        os.path.join(dataset_dir, 'test', 'PNEUMONIA')
    ]
    
    if all(os.path.exists(d) for d in required_dirs):
        print("Датасет уже существует. Пропускаем загрузку.")
        return dataset_dir
    
    # Если данные не найдены, создаем минимальный тестовый датасет
    print("Creating minimal test dataset...")
    for dataset_type in ['train', 'test']:
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(dataset_dir, dataset_type, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(5):  # 5 изображений каждого класса
                if class_name == 'NORMAL':
                    img = np.random.randint(100, 200, (224, 224), dtype=np.uint8)
                else:
                    img = np.random.randint(0, 100, (224, 224), dtype=np.uint8)
                    img[100:150, 100:150] = np.random.randint(150, 255, (50, 50), dtype=np.uint8)
                
                cv2.imwrite(os.path.join(class_dir, f"{class_name.lower()}_{i}.jpeg"), img)
    
    print("Created minimal test dataset with 20 images (5 per class in train and test).")
    return dataset_dir

# Основная функция для подготовки данных
def prepare_data(dataset_dir, output_dir):
    """Полная подготовка данных"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Валидация структуры
    try:
        validate_dataset_structure(dataset_dir)
        # Если структура данных корректна, загружаем данные
        X_train, X_test, y_train, y_test = load_and_prepare_data(dataset_dir)
        print(f"Данные успешно загружены. Тренировочные образцы: {len(X_train)}, Тестовые образцы: {len(X_test)}")
        print(f"Распределение классов: Нормальные={np.sum(y_train==0)}, Пневмония={np.sum(y_train==1)}")
        return X_train, X_test
        
    except FileNotFoundError as e:
        print(f"Ошибка: {str(e)}")
        print("Создаём тестовый датасет...")
        create_test_dataset(dataset_dir)
        return load_and_prepare_data(dataset_dir)
    
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        return None, None

@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        model_type = data.get('model', 'denoiser')
        epochs = int(data.get('epochs', 5))
        
        # Загрузка и проверка данных
        dataset_dir = 'chest_xray'
        X_train, _, _, _ = load_and_prepare_data(dataset_dir)
        
        # Гарантируем размер 224x224
        X_train_resized = [cv2.resize(img, (224, 224)) for img in X_train]
        X_train = np.array(X_train_resized)
        
        # Создание DataLoader
        dataset = DenoisingDataset(X_train, target_size=(224, 224))
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Проверка размеров
        sample_noisy, sample_clean = next(iter(dataloader))
        print(f"Проверка размеров: noisy={sample_noisy.shape}, clean={sample_clean.shape}")
        
        # Инициализация и обучение
        processor = HybridProcessor()
        
        if model_type == 'denoiser':
            processor.neural.train_denoiser(dataloader, epochs)
            torch.save(processor.neural.denoiser.state_dict(), 'weights/denoiser.pth')
        elif model_type == 'unet':
            processor.neural.train_unet(dataloader, epochs)
            torch.save(processor.neural.unet.state_dict(), 'weights/unet.pth')
        
        return jsonify({'status': 'success', 'message': f'Модель {model_type} обучена за {epochs} эпох'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    dataset_dir = 'chest_xray'
    output_dir = 'dataset_processed'
    
    # Проверка и создание структуры папок
    os.makedirs('weights', exist_ok=True)
    
    required_dirs = [
        os.path.join(dataset_dir, 'train', 'NORMAL'),
        os.path.join(dataset_dir, 'train', 'PNEUMONIA'),
        os.path.join(dataset_dir, 'test', 'NORMAL'), 
        os.path.join(dataset_dir, 'test', 'PNEUMONIA')
    ]
    
    # Проверяем наличие уже распакованных данных
    if not all(os.path.exists(d) for d in required_dirs):
        download_and_extract_dataset()
    
    # Подготовка данных
    prepare_data(dataset_dir, output_dir)
    
    # Запуск сервера
    print("Запуск сервера...")
    app.run(host='0.0.0.0', port=5000, debug=True)