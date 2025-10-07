#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lesson15 多模态RAG系统实现模板
解决图像处理、多模态融合和跨模态检索缺失问题

功能特性：
1. 多模态内容处理（文本、图像、音频、视频）
2. 跨模态特征提取和向量化
3. 多模态融合和对齐
4. 跨模态检索和匹配
5. 多模态生成和回答
"""

import logging
import os
import json
import base64
import hashlib
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
from pathlib import Path
import mimetypes

# 图像处理相关库
try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
except ImportError:
    Image = None
    ImageEnhance = None
    ImageFilter = None
    np = None

# 深度学习相关库
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from transformers import (
        CLIPProcessor, CLIPModel,
        BlipProcessor, BlipForConditionalGeneration,
        AutoProcessor, AutoModel
    )
except ImportError:
    torch = None
    nn = None
    transforms = None
    CLIPProcessor = None
    CLIPModel = None
    BlipProcessor = None
    BlipForConditionalGeneration = None
    AutoProcessor = None
    AutoModel = None

# 音频处理相关库
try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa = None
    sf = None

# 视频处理相关库
try:
    import cv2
except ImportError:
    cv2 = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """模态类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"

class ProcessingStatus(Enum):
    """处理状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class MediaContent:
    """媒体内容数据类"""
    content_id: str
    modality: ModalityType
    file_path: Optional[str] = None
    content_data: Optional[bytes] = None
    metadata: Dict = field(default_factory=dict)
    features: Dict = field(default_factory=dict)
    embeddings: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: ProcessingStatus = ProcessingStatus.PENDING
    
@dataclass
class MultimodalQuery:
    """多模态查询"""
    query_id: str
    text_query: Optional[str] = None
    image_query: Optional[str] = None
    audio_query: Optional[str] = None
    video_query: Optional[str] = None
    modality_weights: Dict[str, float] = field(default_factory=dict)
    fusion_strategy: str = "weighted_average"
    
@dataclass
class MultimodalResult:
    """多模态检索结果"""
    content_id: str
    modality: ModalityType
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    combined_score: float = 0.0
    content_preview: Dict = field(default_factory=dict)
    explanation: str = ""

class ImageProcessor:
    """图像处理器"""
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.max_size = (1024, 1024)
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        
        # 加载预训练模型
        self._load_models()
        
        # 图像预处理管道
        if transforms:
            self.transform = transforms.Compose([
                transforms.Resize(self.max_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _load_models(self):
        """加载预训练模型"""
        try:
            if CLIPModel and CLIPProcessor:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model.to(self.device)
                logger.info("CLIP模型加载成功")
            else:
                self.clip_model = None
                self.clip_processor = None
                logger.warning("CLIP模型不可用")
                
            if BlipForConditionalGeneration and BlipProcessor:
                self.blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                self.blip_processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                self.blip_model.to(self.device)
                logger.info("BLIP模型加载成功")
            else:
                self.blip_model = None
                self.blip_processor = None
                logger.warning("BLIP模型不可用")
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.clip_model = None
            self.clip_processor = None
            self.blip_model = None
            self.blip_processor = None
    
    def process_image(self, image_path: str) -> MediaContent:
        """处理图像文件"""
        content_id = self._generate_content_id(image_path)
        
        try:
            # 验证文件格式
            if not self._is_supported_format(image_path):
                raise ValueError(f"不支持的图像格式: {image_path}")
            
            # 加载图像
            if Image:
                image = Image.open(image_path).convert('RGB')
            else:
                raise ImportError("PIL库不可用")
            
            # 创建媒体内容对象
            media_content = MediaContent(
                content_id=content_id,
                modality=ModalityType.IMAGE,
                file_path=image_path,
                status=ProcessingStatus.PROCESSING
            )
            
            # 提取基础元数据
            media_content.metadata = self._extract_image_metadata(image, image_path)
            
            # 图像预处理
            processed_image = self._preprocess_image(image)
            
            # 特征提取
            media_content.features = self._extract_image_features(image, processed_image)
            
            # 生成嵌入向量
            media_content.embeddings = self._generate_image_embeddings(image)
            
            media_content.status = ProcessingStatus.COMPLETED
            logger.info(f"图像处理完成: {image_path}")
            
            return media_content
            
        except Exception as e:
            logger.error(f"图像处理失败 {image_path}: {e}")
            return MediaContent(
                content_id=content_id,
                modality=ModalityType.IMAGE,
                file_path=image_path,
                status=ProcessingStatus.FAILED,
                metadata={'error': str(e)}
            )
    
    def _is_supported_format(self, file_path: str) -> bool:
        """检查是否为支持的图像格式"""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def _extract_image_metadata(self, image: 'Image.Image', file_path: str) -> Dict:
        """提取图像元数据"""
        metadata = {
            'width': image.width,
            'height': image.height,
            'mode': image.mode,
            'format': image.format,
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'aspect_ratio': image.width / image.height if image.height > 0 else 0
        }
        
        # 提取EXIF信息
        try:
            exif = image._getexif()
            if exif:
                metadata['exif'] = dict(exif)
        except:
            pass
        
        return metadata
    
    def _preprocess_image(self, image: 'Image.Image') -> 'Image.Image':
        """图像预处理"""
        # 调整大小
        if image.size[0] > self.max_size[0] or image.size[1] > self.max_size[1]:
            image = image.resize(self.max_size, Image.Resampling.LANCZOS)
        
        # 图像增强
        if ImageEnhance:
            # 对比度增强
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            # 锐化
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
        
        return image
    
    def _extract_image_features(self, original_image: 'Image.Image', 
                               processed_image: 'Image.Image') -> Dict:
        """提取图像特征"""
        features = {}
        
        if np:
            # 转换为numpy数组
            img_array = np.array(processed_image)
            
            # 颜色特征
            features['color_histogram'] = self._compute_color_histogram(img_array)
            features['dominant_colors'] = self._extract_dominant_colors(img_array)
            
            # 纹理特征
            features['texture_features'] = self._compute_texture_features(img_array)
            
            # 形状特征
            features['shape_features'] = self._compute_shape_features(img_array)
        
        return features
    
    def _compute_color_histogram(self, img_array: 'np.ndarray') -> Dict:
        """计算颜色直方图"""
        if not np:
            return {}
        
        # RGB直方图
        hist_r = np.histogram(img_array[:, :, 0], bins=32, range=(0, 256))[0]
        hist_g = np.histogram(img_array[:, :, 1], bins=32, range=(0, 256))[0]
        hist_b = np.histogram(img_array[:, :, 2], bins=32, range=(0, 256))[0]
        
        return {
            'red_histogram': hist_r.tolist(),
            'green_histogram': hist_g.tolist(),
            'blue_histogram': hist_b.tolist()
        }
    
    def _extract_dominant_colors(self, img_array: 'np.ndarray', k: int = 5) -> List[List[int]]:
        """提取主要颜色"""
        if not np:
            return []
        
        # 简化的主要颜色提取（使用K-means的简化版本）
        pixels = img_array.reshape(-1, 3)
        
        # 随机采样减少计算量
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        # 简单的颜色聚类
        dominant_colors = []
        for _ in range(k):
            mean_color = np.mean(pixels, axis=0).astype(int)
            dominant_colors.append(mean_color.tolist())
            
            # 移除相似的像素
            distances = np.sum((pixels - mean_color) ** 2, axis=1)
            mask = distances > np.percentile(distances, 20)
            pixels = pixels[mask]
            
            if len(pixels) < 100:
                break
        
        return dominant_colors
    
    def _compute_texture_features(self, img_array: 'np.ndarray') -> Dict:
        """计算纹理特征"""
        if not np:
            return {}
        
        # 转换为灰度图
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        
        # 简单的纹理特征
        features = {
            'mean_intensity': float(np.mean(gray)),
            'std_intensity': float(np.std(gray)),
            'contrast': float(np.std(gray) / np.mean(gray)) if np.mean(gray) > 0 else 0.0
        }
        
        return features
    
    def _compute_shape_features(self, img_array: 'np.ndarray') -> Dict:
        """计算形状特征"""
        if not np:
            return {}
        
        # 边缘检测（简化版本）
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
        
        # 简单的边缘检测
        edges_x = np.abs(np.diff(gray, axis=1))
        edges_y = np.abs(np.diff(gray, axis=0))
        
        features = {
            'edge_density_x': float(np.mean(edges_x)),
            'edge_density_y': float(np.mean(edges_y)),
            'total_edge_density': float(np.mean(edges_x) + np.mean(edges_y))
        }
        
        return features
    
    def _generate_image_embeddings(self, image: 'Image.Image') -> Dict:
        """生成图像嵌入向量"""
        embeddings = {}
        
        try:
            # CLIP嵌入
            if self.clip_model and self.clip_processor:
                inputs = self.clip_processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    embeddings['clip'] = image_features.cpu().numpy().flatten().tolist()
            
            # 图像描述生成
            if self.blip_model and self.blip_processor:
                inputs = self.blip_processor(image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = self.blip_model.generate(**inputs, max_length=50)
                    caption = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
                    embeddings['caption'] = caption
                    
                    # 使用CLIP对生成的描述进行编码
                    if self.clip_model and self.clip_processor:
                        text_inputs = self.clip_processor(text=[caption], return_tensors="pt")
                        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                        text_features = self.clip_model.get_text_features(**text_inputs)
                        embeddings['caption_clip'] = text_features.cpu().numpy().flatten().tolist()
        
        except Exception as e:
            logger.error(f"生成图像嵌入失败: {e}")
        
        return embeddings
    
    def _generate_content_id(self, file_path: str) -> str:
        """生成内容ID"""
        return hashlib.md5(f"image_{file_path}_{datetime.now().isoformat()}".encode()).hexdigest()

class AudioProcessor:
    """音频处理器"""
    
    def __init__(self):
        self.supported_formats = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        self.sample_rate = 16000
        self.max_duration = 300  # 5分钟
    
    def process_audio(self, audio_path: str) -> MediaContent:
        """处理音频文件"""
        content_id = self._generate_content_id(audio_path)
        
        try:
            if not self._is_supported_format(audio_path):
                raise ValueError(f"不支持的音频格式: {audio_path}")
            
            if not librosa:
                raise ImportError("librosa库不可用")
            
            # 加载音频
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # 限制音频长度
            if len(audio_data) > self.sample_rate * self.max_duration:
                audio_data = audio_data[:self.sample_rate * self.max_duration]
            
            media_content = MediaContent(
                content_id=content_id,
                modality=ModalityType.AUDIO,
                file_path=audio_path,
                status=ProcessingStatus.PROCESSING
            )
            
            # 提取元数据
            media_content.metadata = self._extract_audio_metadata(audio_data, sr, audio_path)
            
            # 特征提取
            media_content.features = self._extract_audio_features(audio_data, sr)
            
            # 生成嵌入向量
            media_content.embeddings = self._generate_audio_embeddings(audio_data, sr)
            
            media_content.status = ProcessingStatus.COMPLETED
            logger.info(f"音频处理完成: {audio_path}")
            
            return media_content
            
        except Exception as e:
            logger.error(f"音频处理失败 {audio_path}: {e}")
            return MediaContent(
                content_id=content_id,
                modality=ModalityType.AUDIO,
                file_path=audio_path,
                status=ProcessingStatus.FAILED,
                metadata={'error': str(e)}
            )
    
    def _is_supported_format(self, file_path: str) -> bool:
        """检查是否为支持的音频格式"""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def _extract_audio_metadata(self, audio_data: 'np.ndarray', sr: int, file_path: str) -> Dict:
        """提取音频元数据"""
        duration = len(audio_data) / sr
        
        metadata = {
            'duration': duration,
            'sample_rate': sr,
            'channels': 1,  # librosa默认加载为单声道
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'samples': len(audio_data)
        }
        
        return metadata
    
    def _extract_audio_features(self, audio_data: 'np.ndarray', sr: int) -> Dict:
        """提取音频特征"""
        features = {}
        
        try:
            # MFCC特征
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
            features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
            
            # 频谱质心
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # 零交叉率
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # 色度特征
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
            
            # 节拍和节奏
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
            features['tempo'] = float(tempo)
            features['beat_count'] = len(beats)
            
        except Exception as e:
            logger.error(f"音频特征提取失败: {e}")
        
        return features
    
    def _generate_audio_embeddings(self, audio_data: 'np.ndarray', sr: int) -> Dict:
        """生成音频嵌入向量"""
        embeddings = {}
        
        try:
            # 使用MFCC作为基础嵌入
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
            # 统计特征作为嵌入
            mfcc_features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.max(mfccs, axis=1),
                np.min(mfccs, axis=1)
            ])
            embeddings['mfcc_statistical'] = mfcc_features.tolist()
            
            # 频谱特征嵌入
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            spectral_features = np.concatenate([
                np.mean(magnitude, axis=1),
                np.std(magnitude, axis=1)
            ])
            embeddings['spectral'] = spectral_features[:128].tolist()  # 限制维度
            
        except Exception as e:
            logger.error(f"生成音频嵌入失败: {e}")
        
        return embeddings
    
    def _generate_content_id(self, file_path: str) -> str:
        """生成内容ID"""
        return hashlib.md5(f"audio_{file_path}_{datetime.now().isoformat()}".encode()).hexdigest()

class VideoProcessor:
    """视频处理器"""
    
    def __init__(self):
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        self.max_duration = 600  # 10分钟
        self.frame_sample_rate = 1  # 每秒采样1帧
        
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
    
    def process_video(self, video_path: str) -> MediaContent:
        """处理视频文件"""
        content_id = self._generate_content_id(video_path)
        
        try:
            if not self._is_supported_format(video_path):
                raise ValueError(f"不支持的视频格式: {video_path}")
            
            if not cv2:
                raise ImportError("OpenCV库不可用")
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            
            media_content = MediaContent(
                content_id=content_id,
                modality=ModalityType.VIDEO,
                file_path=video_path,
                status=ProcessingStatus.PROCESSING
            )
            
            # 提取元数据
            media_content.metadata = self._extract_video_metadata(cap, video_path)
            
            # 提取关键帧
            frames = self._extract_key_frames(cap)
            
            # 特征提取
            media_content.features = self._extract_video_features(frames, media_content.metadata)
            
            # 生成嵌入向量
            media_content.embeddings = self._generate_video_embeddings(frames)
            
            cap.release()
            media_content.status = ProcessingStatus.COMPLETED
            logger.info(f"视频处理完成: {video_path}")
            
            return media_content
            
        except Exception as e:
            logger.error(f"视频处理失败 {video_path}: {e}")
            return MediaContent(
                content_id=content_id,
                modality=ModalityType.VIDEO,
                file_path=video_path,
                status=ProcessingStatus.FAILED,
                metadata={'error': str(e)}
            )
    
    def _is_supported_format(self, file_path: str) -> bool:
        """检查是否为支持的视频格式"""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def _extract_video_metadata(self, cap: 'cv2.VideoCapture', file_path: str) -> Dict:
        """提取视频元数据"""
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        metadata = {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration,
            'aspect_ratio': width / height if height > 0 else 0,
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
        
        return metadata
    
    def _extract_key_frames(self, cap: 'cv2.VideoCapture', max_frames: int = 30) -> List['np.ndarray']:
        """提取关键帧"""
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or frame_count <= 0:
            return frames
        
        # 计算采样间隔
        interval = max(1, int(fps / self.frame_sample_rate))
        
        frame_indices = list(range(0, frame_count, interval))[:max_frames]
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # 转换颜色空间（BGR到RGB）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        return frames
    
    def _extract_video_features(self, frames: List['np.ndarray'], metadata: Dict) -> Dict:
        """提取视频特征"""
        features = {
            'frame_count': len(frames),
            'temporal_features': {},
            'visual_features': {}
        }
        
        if not frames:
            return features
        
        try:
            # 时间特征
            features['temporal_features'] = {
                'duration': metadata.get('duration', 0),
                'fps': metadata.get('fps', 0),
                'frame_sampling_rate': len(frames) / metadata.get('duration', 1)
            }
            
            # 视觉特征统计
            frame_features = []
            for frame in frames:
                if Image:
                    # 转换为PIL图像
                    pil_frame = Image.fromarray(frame)
                    # 使用图像处理器提取特征
                    img_features = self.image_processor._extract_image_features(pil_frame, pil_frame)
                    frame_features.append(img_features)
            
            if frame_features:
                # 聚合帧特征
                features['visual_features'] = self._aggregate_frame_features(frame_features)
            
        except Exception as e:
            logger.error(f"视频特征提取失败: {e}")
        
        return features
    
    def _aggregate_frame_features(self, frame_features: List[Dict]) -> Dict:
        """聚合帧特征"""
        aggregated = {}
        
        if not frame_features:
            return aggregated
        
        # 聚合颜色特征
        if 'color_histogram' in frame_features[0]:
            color_hists = [f.get('color_histogram', {}) for f in frame_features]
            aggregated['avg_color_histogram'] = self._average_histograms(color_hists)
        
        # 聚合纹理特征
        texture_features = [f.get('texture_features', {}) for f in frame_features]
        if texture_features:
            aggregated['avg_texture'] = self._average_texture_features(texture_features)
        
        return aggregated
    
    def _average_histograms(self, histograms: List[Dict]) -> Dict:
        """平均颜色直方图"""
        if not histograms or not histograms[0]:
            return {}
        
        avg_hist = {}
        for channel in ['red_histogram', 'green_histogram', 'blue_histogram']:
            if channel in histograms[0]:
                channel_hists = [h.get(channel, []) for h in histograms if h.get(channel)]
                if channel_hists and np:
                    avg_hist[channel] = np.mean(channel_hists, axis=0).tolist()
        
        return avg_hist
    
    def _average_texture_features(self, texture_features: List[Dict]) -> Dict:
        """平均纹理特征"""
        if not texture_features:
            return {}
        
        avg_texture = {}
        feature_keys = ['mean_intensity', 'std_intensity', 'contrast']
        
        for key in feature_keys:
            values = [f.get(key, 0) for f in texture_features if key in f]
            if values:
                avg_texture[key] = float(np.mean(values)) if np else sum(values) / len(values)
        
        return avg_texture
    
    def _generate_video_embeddings(self, frames: List['np.ndarray']) -> Dict:
        """生成视频嵌入向量"""
        embeddings = {}
        
        if not frames:
            return embeddings
        
        try:
            # 使用关键帧生成嵌入
            frame_embeddings = []
            
            for frame in frames[:10]:  # 限制帧数
                if Image and self.image_processor.clip_model:
                    pil_frame = Image.fromarray(frame)
                    frame_emb = self.image_processor._generate_image_embeddings(pil_frame)
                    if 'clip' in frame_emb:
                        frame_embeddings.append(frame_emb['clip'])
            
            if frame_embeddings and np:
                # 平均池化
                embeddings['clip_average'] = np.mean(frame_embeddings, axis=0).tolist()
                
                # 最大池化
                embeddings['clip_max'] = np.max(frame_embeddings, axis=0).tolist()
                
                # 时间序列特征
                if len(frame_embeddings) > 1:
                    # 计算帧间差异
                    frame_diffs = []
                    for i in range(1, len(frame_embeddings)):
                        diff = np.array(frame_embeddings[i]) - np.array(frame_embeddings[i-1])
                        frame_diffs.append(np.linalg.norm(diff))
                    
                    embeddings['temporal_dynamics'] = {
                        'mean_frame_diff': float(np.mean(frame_diffs)),
                        'std_frame_diff': float(np.std(frame_diffs)),
                        'max_frame_diff': float(np.max(frame_diffs))
                    }
        
        except Exception as e:
            logger.error(f"生成视频嵌入失败: {e}")
        
        return embeddings
    
    def _generate_content_id(self, file_path: str) -> str:
        """生成内容ID"""
        return hashlib.md5(f"video_{file_path}_{datetime.now().isoformat()}".encode()).hexdigest()

class MultimodalFusionEngine:
    """多模态融合引擎"""
    
    def __init__(self):
        self.fusion_strategies = {
            'weighted_average': self._weighted_average_fusion,
            'attention_fusion': self._attention_fusion,
            'concatenation': self._concatenation_fusion,
            'max_pooling': self._max_pooling_fusion
        }
    
    def fuse_embeddings(self, embeddings_dict: Dict[str, List[float]], 
                       weights: Dict[str, float] = None,
                       strategy: str = 'weighted_average') -> List[float]:
        """融合多模态嵌入向量"""
        if not embeddings_dict:
            return []
        
        if strategy not in self.fusion_strategies:
            strategy = 'weighted_average'
        
        return self.fusion_strategies[strategy](embeddings_dict, weights)
    
    def _weighted_average_fusion(self, embeddings_dict: Dict[str, List[float]], 
                                weights: Dict[str, float] = None) -> List[float]:
        """加权平均融合"""
        if not embeddings_dict or not np:
            return []
        
        # 默认权重
        if weights is None:
            weights = {k: 1.0 for k in embeddings_dict.keys()}
        
        # 标准化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # 找到最大维度
        max_dim = max(len(emb) for emb in embeddings_dict.values())
        
        # 融合嵌入
        fused_embedding = np.zeros(max_dim)
        
        for modality, embedding in embeddings_dict.items():
            weight = weights.get(modality, 0.0)
            if weight > 0 and embedding:
                # 填充或截断到统一维度
                padded_emb = np.array(embedding[:max_dim])
                if len(padded_emb) < max_dim:
                    padded_emb = np.pad(padded_emb, (0, max_dim - len(padded_emb)))
                
                fused_embedding += weight * padded_emb
        
        return fused_embedding.tolist()
    
    def _attention_fusion(self, embeddings_dict: Dict[str, List[float]], 
                         weights: Dict[str, float] = None) -> List[float]:
        """注意力机制融合"""
        if not embeddings_dict or not np:
            return []
        
        # 简化的注意力机制
        embeddings_list = list(embeddings_dict.values())
        modalities = list(embeddings_dict.keys())
        
        if not embeddings_list:
            return []
        
        # 计算注意力权重
        attention_weights = []
        for i, emb in enumerate(embeddings_list):
            if emb:
                # 使用嵌入的L2范数作为注意力分数
                attention_score = np.linalg.norm(emb)
                attention_weights.append(attention_score)
            else:
                attention_weights.append(0.0)
        
        # 归一化注意力权重
        total_attention = sum(attention_weights)
        if total_attention > 0:
            attention_weights = [w / total_attention for w in attention_weights]
        
        # 应用注意力权重
        attention_dict = {modalities[i]: attention_weights[i] for i in range(len(modalities))}
        
        return self._weighted_average_fusion(embeddings_dict, attention_dict)
    
    def _concatenation_fusion(self, embeddings_dict: Dict[str, List[float]], 
                             weights: Dict[str, float] = None) -> List[float]:
        """拼接融合"""
        fused_embedding = []
        
        # 按模态顺序拼接
        for modality in sorted(embeddings_dict.keys()):
            embedding = embeddings_dict[modality]
            if embedding:
                fused_embedding.extend(embedding)
        
        return fused_embedding
    
    def _max_pooling_fusion(self, embeddings_dict: Dict[str, List[float]], 
                           weights: Dict[str, float] = None) -> List[float]:
        """最大池化融合"""
        if not embeddings_dict or not np:
            return []
        
        # 找到最大维度
        max_dim = max(len(emb) for emb in embeddings_dict.values() if emb)
        
        if max_dim == 0:
            return []
        
        # 收集所有嵌入
        all_embeddings = []
        for embedding in embeddings_dict.values():
            if embedding:
                # 填充或截断到统一维度
                padded_emb = np.array(embedding[:max_dim])
                if len(padded_emb) < max_dim:
                    padded_emb = np.pad(padded_emb, (0, max_dim - len(padded_emb)))
                all_embeddings.append(padded_emb)
        
        if all_embeddings:
            # 最大池化
            fused_embedding = np.max(all_embeddings, axis=0)
            return fused_embedding.tolist()
        
        return []

class MultimodalRetriever:
    """多模态检索器"""
    
    def __init__(self):
        self.content_database = {}  # content_id -> MediaContent
        self.fusion_engine = MultimodalFusionEngine()
        self._lock = threading.RLock()
    
    def add_content(self, content: MediaContent) -> None:
        """添加内容到数据库"""
        with self._lock:
            self.content_database[content.content_id] = content
            logger.info(f"添加内容: {content.content_id} ({content.modality.value})")
    
    def search(self, query: MultimodalQuery, top_k: int = 10) -> List[MultimodalResult]:
        """多模态检索"""
        with self._lock:
            results = []
            
            for content_id, content in self.content_database.items():
                if content.status != ProcessingStatus.COMPLETED:
                    continue
                
                # 计算相似度分数
                similarity_scores = self._calculate_similarity(query, content)
                
                # 融合分数
                combined_score = self._combine_scores(similarity_scores, query.modality_weights)
                
                if combined_score > 0:
                    result = MultimodalResult(
                        content_id=content_id,
                        modality=content.modality,
                        similarity_scores=similarity_scores,
                        combined_score=combined_score,
                        content_preview=self._generate_content_preview(content),
                        explanation=self._generate_explanation(similarity_scores, content)
                    )
                    results.append(result)
            
            # 排序并返回top-k结果
            results.sort(key=lambda x: x.combined_score, reverse=True)
            return results[:top_k]
    
    def _calculate_similarity(self, query: MultimodalQuery, content: MediaContent) -> Dict[str, float]:
        """计算相似度分数"""
        similarity_scores = {}
        
        # 文本相似度
        if query.text_query and 'caption' in content.embeddings:
            text_sim = self._text_similarity(query.text_query, content.embeddings['caption'])
            similarity_scores['text'] = text_sim
        
        # 图像相似度
        if query.image_query and 'clip' in content.embeddings:
            # 这里简化处理，实际应该对查询图像进行编码
            image_sim = 0.5  # 占位符
            similarity_scores['image'] = image_sim
        
        # 基于内容类型的相似度
        if content.modality == ModalityType.IMAGE and 'clip' in content.embeddings:
            similarity_scores['visual'] = 0.8  # 占位符
        elif content.modality == ModalityType.AUDIO and 'mfcc_statistical' in content.embeddings:
            similarity_scores['audio'] = 0.6  # 占位符
        elif content.modality == ModalityType.VIDEO and 'clip_average' in content.embeddings:
            similarity_scores['video'] = 0.7  # 占位符
        
        return similarity_scores
    
    def _text_similarity(self, query_text: str, content_caption: str) -> float:
        """计算文本相似度"""
        # 简化的文本相似度计算
        query_words = set(query_text.lower().split())
        caption_words = set(content_caption.lower().split())
        
        if not query_words or not caption_words:
            return 0.0
        
        intersection = query_words.intersection(caption_words)
        union = query_words.union(caption_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _combine_scores(self, similarity_scores: Dict[str, float], 
                       modality_weights: Dict[str, float]) -> float:
        """组合相似度分数"""
        if not similarity_scores:
            return 0.0
        
        # 默认权重
        if not modality_weights:
            modality_weights = {k: 1.0 for k in similarity_scores.keys()}
        
        # 加权平均
        weighted_sum = 0.0
        total_weight = 0.0
        
        for modality, score in similarity_scores.items():
            weight = modality_weights.get(modality, 1.0)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_content_preview(self, content: MediaContent) -> Dict:
        """生成内容预览"""
        preview = {
            'content_id': content.content_id,
            'modality': content.modality.value,
            'file_path': content.file_path,
            'metadata': content.metadata
        }
        
        # 添加模态特定的预览信息
        if content.modality == ModalityType.IMAGE:
            preview['image_info'] = {
                'dimensions': f"{content.metadata.get('width', 0)}x{content.metadata.get('height', 0)}",
                'format': content.metadata.get('format', 'unknown')
            }
            if 'caption' in content.embeddings:
                preview['caption'] = content.embeddings['caption']
        
        elif content.modality == ModalityType.AUDIO:
            preview['audio_info'] = {
                'duration': f"{content.metadata.get('duration', 0):.2f}s",
                'sample_rate': content.metadata.get('sample_rate', 0)
            }
        
        elif content.modality == ModalityType.VIDEO:
            preview['video_info'] = {
                'duration': f"{content.metadata.get('duration', 0):.2f}s",
                'dimensions': f"{content.metadata.get('width', 0)}x{content.metadata.get('height', 0)}",
                'fps': content.metadata.get('fps', 0)
            }
        
        return preview
    
    def _generate_explanation(self, similarity_scores: Dict[str, float], 
                             content: MediaContent) -> str:
        """生成匹配解释"""
        explanations = []
        
        for modality, score in similarity_scores.items():
            if score > 0.5:
                explanations.append(f"{modality}相似度: {score:.2f}")
        
        if content.modality == ModalityType.IMAGE and 'caption' in content.embeddings:
            explanations.append(f"图像描述: {content.embeddings['caption']}")
        
        return "; ".join(explanations) if explanations else "基础匹配"
    
    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        with self._lock:
            stats = {
                'total_content': len(self.content_database),
                'by_modality': {},
                'by_status': {}
            }
            
            for content in self.content_database.values():
                # 按模态统计
                modality = content.modality.value
                stats['by_modality'][modality] = stats['by_modality'].get(modality, 0) + 1
                
                # 按状态统计
                status = content.status.value
                stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            
            return stats

class MultimodalRAGSystem:
    """多模态RAG系统"""
    
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        self.retriever = MultimodalRetriever()
        self._lock = threading.RLock()
    
    def add_content_from_file(self, file_path: str) -> Optional[MediaContent]:
        """从文件添加内容"""
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return None
        
        # 检测文件类型
        mime_type, _ = mimetypes.guess_type(file_path)
        
        try:
            if mime_type and mime_type.startswith('image/'):
                content = self.image_processor.process_image(file_path)
            elif mime_type and mime_type.startswith('audio/'):
                content = self.audio_processor.process_audio(file_path)
            elif mime_type and mime_type.startswith('video/'):
                content = self.video_processor.process_video(file_path)
            else:
                logger.warning(f"不支持的文件类型: {file_path} ({mime_type})")
                return None
            
            if content.status == ProcessingStatus.COMPLETED:
                self.retriever.add_content(content)
                return content
            else:
                logger.error(f"内容处理失败: {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"添加内容失败 {file_path}: {e}")
            return None
    
    def search_multimodal(self, text_query: str = None, image_path: str = None,
                         modality_weights: Dict[str, float] = None,
                         top_k: int = 10) -> List[MultimodalResult]:
        """多模态搜索"""
        query = MultimodalQuery(
            query_id=hashlib.md5(f"{text_query}_{image_path}_{datetime.now().isoformat()}".encode()).hexdigest(),
            text_query=text_query,
            image_query=image_path,
            modality_weights=modality_weights or {}
        )
        
        return self.retriever.search(query, top_k)
    
    def batch_add_content(self, file_paths: List[str]) -> Dict[str, Any]:
        """批量添加内容"""
        results = {
            'successful': [],
            'failed': [],
            'total': len(file_paths)
        }
        
        for file_path in file_paths:
            content = self.add_content_from_file(file_path)
            if content:
                results['successful'].append({
                    'file_path': file_path,
                    'content_id': content.content_id,
                    'modality': content.modality.value
                })
            else:
                results['failed'].append(file_path)
        
        return results
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        status = {
            'processors': {
                'image': {
                    'available': self.image_processor.clip_model is not None,
                    'models': {
                        'clip': self.image_processor.clip_model is not None,
                        'blip': self.image_processor.blip_model is not None
                    }
                },
                'audio': {
                    'available': librosa is not None,
                    'supported_formats': list(self.audio_processor.supported_formats)
                },
                'video': {
                    'available': cv2 is not None,
                    'supported_formats': list(self.video_processor.supported_formats)
                }
            },
            'database': self.retriever.get_statistics(),
            'fusion_strategies': list(self.retriever.fusion_engine.fusion_strategies.keys())
        }
        
        return status

def main():
    """示例用法"""
    # 创建多模态RAG系统
    rag_system = MultimodalRAGSystem()
    
    print("多模态RAG系统测试\n" + "="*50)
    
    # 显示系统状态
    status = rag_system.get_system_status()
    print("系统状态:")
    for component, info in status['processors'].items():
        print(f"  {component}: {'可用' if info['available'] else '不可用'}")
    
    # 模拟添加内容（实际使用时需要真实的文件路径）
    test_files = [
        # "path/to/image.jpg",
        # "path/to/audio.wav",
        # "path/to/video.mp4"
    ]
    
    if test_files:
        print(f"\n批量添加内容...")
        batch_results = rag_system.batch_add_content(test_files)
        print(f"成功: {len(batch_results['successful'])}, 失败: {len(batch_results['failed'])}")
        
        # 多模态搜索
        print(f"\n多模态搜索测试...")
        search_results = rag_system.search_multimodal(
            text_query="机器学习算法",
            modality_weights={'text': 0.6, 'visual': 0.4},
            top_k=5
        )
        
        print(f"找到 {len(search_results)} 个结果:")
        for i, result in enumerate(search_results, 1):
            print(f"  {i}. {result.content_id} ({result.modality.value}) - 分数: {result.combined_score:.3f}")
            print(f"     解释: {result.explanation}")
    else:
        print("\n没有提供测试文件，跳过内容添加和搜索测试")
    
    # 显示数据库统计
    db_stats = rag_system.retriever.get_statistics()
    print(f"\n数据库统计:")
    print(f"  总内容数: {db_stats['total_content']}")
    print(f"  按模态分布: {db_stats['by_modality']}")
    print(f"  按状态分布: {db_stats['by_status']}")

if __name__ == "__main__":
    main()