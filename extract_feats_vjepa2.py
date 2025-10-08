#!/usr/bin/env python3
"""
V-JEPA2 Feature Extraction Module
Extracts features from video sequences using V-JEPA2 self-supervised models
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict, Any
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import cv2
from transformers import AutoVideoProcessor, AutoModel

# Optional torchcodec import
try:
    from torchcodec.decoders import VideoDecoder
    HAS_TORCHCODEC = True
except ImportError:
    VideoDecoder = None
    HAS_TORCHCODEC = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# V-JEPA2 Model configurations (Hugging Face repositories)
VJEPA2_MODELS = {
    'vjepa2_vitl16': 'facebook/vjepa2-vitl-fpc64-256'
}

# Model feature dimensions
VJEPA2_FEATURE_DIMS = {
    'vjepa2_vitl16': 1024
}

# Default model
DEFAULT_VJEPA2_MODEL = 'vjepa2_vitl16'

# Image preprocessing constants
VJEPA2_MEAN = [0.485, 0.456, 0.406]
VJEPA2_STD = [0.229, 0.224, 0.225]
VJEPA2_IMAGE_SIZE = 224


class VJEPA2FeatureExtractor:
    """V-JEPA2 based feature extractor for video sequences"""
    
    def __init__(self, model_name: str = DEFAULT_VJEPA2_MODEL, device: str = "cuda", 
                 is_main_process: bool = True):
        """
        Initialize V-JEPA2 feature extractor using Hugging Face transformers
        
        Args:
            model_name: V-JEPA2 model name
            device: Device to use ('cuda' or 'cpu')
            is_main_process: Whether this is the main process (for distributed training)
        """
        self.model_name = model_name
        self.device = device
        self.is_main_process = is_main_process
        
        if self.is_main_process:
            print(f"Initializing V-JEPA2 feature extractor with model: {model_name}")
        
        # Load model and processor from Hugging Face
        self.model, self.processor = self._load_model()
        
        if self.is_main_process:
            print(f"✓ V-JEPA2 {model_name} loaded successfully on {device}")
    
    def _load_model(self) -> Tuple[AutoModel, AutoVideoProcessor]:
        """Load V-JEPA2 model and processor from Hugging Face"""
        try:
            print(f"Loading V-JEPA2 model from: {self.model_name}")
            hf_repo = VJEPA2_MODELS[self.model_name]
            
            if self.is_main_process:
                print(f"Loading V-JEPA2 model from: {hf_repo}")
            
            # Load model and processor from Hugging Face
            model = AutoModel.from_pretrained(hf_repo)
            processor = AutoVideoProcessor.from_pretrained(hf_repo)
            
            # Move model to device
            model = model.to(self.device)
            model.eval()
            
            return model, processor
                
        except Exception as e:
            if self.is_main_process:
                print(f"Error loading V-JEPA2 model: {e}")
                print("Make sure you have transformers and torchcodec installed:")
                print("pip install transformers torchcodec")
            raise e
    
    def _prepare_video_from_frames(self, frame_paths: List[str], max_frames: int = 64) -> torch.Tensor:
        """
        Prepare video tensor from frame paths using V-JEPA2 format
        
        Args:
            frame_paths: List of frame file paths
            max_frames: Maximum number of frames (V-JEPA2 default is 64)
            
        Returns:
            Video tensor of shape (T, C, H, W)
        """
        # Sample frames if too many
        if len(frame_paths) > max_frames:
            indices = np.linspace(0, len(frame_paths)-1, max_frames, dtype=int)
            frame_paths = [frame_paths[i] for i in indices]
        
        frames = []
        for frame_path in frame_paths:
            try:
                # Load image and convert to RGB
                image = Image.open(frame_path).convert('RGB')
                # Convert to numpy array (H, W, C)
                frame_array = np.array(image)
                frames.append(frame_array)
            except Exception as e:
                if self.is_main_process:
                    print(f"Warning: Skipping frame {frame_path}: {e}")
                continue
        
        if not frames:
            return None
        
        # Convert to tensor format expected by V-JEPA2: (T, C, H, W)
        video_array = np.stack(frames, axis=0)  # (T, H, W, C)
        video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        
        return video_tensor
    
    def extract_image_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract features from a single image using V-JEPA2
        Note: V-JEPA2 is designed for video, so we create a single-frame video
        """
        try:
            # Create single-frame video
            video_tensor = self._prepare_video_from_frames([image_path], max_frames=1)
            if video_tensor is None:
                return None
            
            # Process with V-JEPA2 processor and model
            video_processed = self.processor(video_tensor, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                features = self.model.get_vision_features(**video_processed)
            
            return features.cpu().numpy()
            
        except Exception as e:
            if self.is_main_process:
                print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def extract_video_features(self, frame_paths: List[str], 
                              max_frames: int = 64) -> Optional[np.ndarray]:
        """
        Extract features from video frames using V-JEPA2
        
        Args:
            frame_paths: List of frame file paths
            max_frames: Maximum frames to process (V-JEPA2 default is 64)
            
        Returns:
            Feature array from V-JEPA2 model
        """
        try:
            # Prepare video tensor
            video_tensor = self._prepare_video_from_frames(frame_paths, max_frames)
            if video_tensor is None:
                return None
            
            # Process with V-JEPA2 processor and model
            video_processed = self.processor(video_tensor, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                features = self.model.get_vision_features(**video_processed)
            
            return features
            
        except Exception as e:
            if self.is_main_process:
                print(f"Error extracting video features: {e}")
            return None
    
    
    def extract_video_sequential(self, input_dir: str, gt_dir: str, 
                                input_files: List[str], gt_files: List[str],
                                resize_mode: str = "standard") -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract features from video frames sequentially using V-JEPA2
        
        Args:
            input_dir: Directory containing input frames
            gt_dir: Directory containing ground truth frames
            input_files: List of input frame filenames
            gt_files: List of ground truth frame filenames
            resize_mode: Resize mode (not used for V-JEPA2, kept for compatibility)
            
        Returns:
            Tuple of (video_features, video_labels)
        """
        video_features = []
        video_labels = []
        
        # Process frames in chunks for memory efficiency (V-JEPA2 processes 64 frames at once)
        chunk_size = 64
        
        for i in tqdm(range(0, len(input_files), chunk_size), 
                     desc="Extracting V-JEPA2 features", disable=not self.is_main_process):
            
            chunk_input = input_files[i:i+chunk_size]
            chunk_gt = gt_files[i:i+chunk_size]
            
            # Get full paths
            chunk_input_paths = [os.path.join(input_dir, f) for f in chunk_input]
            chunk_gt_paths = [os.path.join(gt_dir, f) for f in chunk_gt]
            
            # Extract V-JEPA2 features from chunk (returns single feature vector for the video chunk)
            chunk_features = self.extract_video_features(chunk_input_paths, max_frames=len(chunk_input))
            
            if chunk_features is not None:
                # Process ground truth labels for each frame
                chunk_labels = []
                for gt_path in chunk_gt_paths:
                    try:
                        gt_image = Image.open(gt_path).convert('L')  # Grayscale
                        gt_array = np.array(gt_image)
                        # Binary threshold (assuming 255 = foreground, 0 = background)
                        gt_binary = (gt_array > 127).astype(np.float32)
                        chunk_labels.append(gt_binary)
                    except Exception as e:
                        if self.is_main_process:
                            print(f"Warning: Error loading GT {gt_path}: {e}")
                        continue
                
                # V-JEPA2 returns one feature vector per video chunk, replicate for each frame
                if len(chunk_labels) > 0:
                    for label in chunk_labels:
                        video_features.append(chunk_features)  # Same features for all frames in chunk
                        video_labels.append(label)
        
        return video_features, video_labels
    
    def extract_video_batch(self, input_dir: str, gt_dir: str,
                           input_files: List[str], gt_files: List[str],
                           resize_mode: str = "standard", batch_size: int = 4) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract features with batching (same as sequential for V-JEPA2)
        """
        return self.extract_video_sequential(input_dir, gt_dir, input_files, gt_files, resize_mode)


def load_vjepa2_model(model_name: str = DEFAULT_VJEPA2_MODEL, device: str = "cuda",
                      is_main_process: bool = True) -> VJEPA2FeatureExtractor:
    """
    Load V-JEPA2 feature extractor
    
    Args:
        model_name: V-JEPA2 model name
        device: Device to use
        is_main_process: Whether this is the main process
        
    Returns:
        VJEPA2FeatureExtractor instance
    """
    return VJEPA2FeatureExtractor(model_name, device, is_main_process)


def get_vjepa2_feature_dim(model_name: str) -> int:
    """Get feature dimension for V-JEPA2 model"""
    return VJEPA2_FEATURE_DIMS.get(model_name, 768)


if __name__ == "__main__":
    # Test V-JEPA2 feature extraction
    print("Testing V-JEPA2 Feature Extraction with Hugging Face transformers")
    print("Required packages: transformers, torchcodec")
    
    # Test with sample images if available
    test_dir = "data/images"
    if os.path.exists(test_dir):
        try:
            extractor = load_vjepa2_model()
            
            # Test single image
            image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png'))]
            if image_files:
                test_image = os.path.join(test_dir, image_files[0])
                features = extractor.extract_image_features(test_image)
                if features is not None:
                    print(f"✓ Extracted image features shape: {features.shape}")
                else:
                    print("✗ Failed to extract image features")
            
            # Test video frames
            if len(image_files) >= 3:
                frame_paths = [os.path.join(test_dir, f) for f in image_files[:3]]
                video_features = extractor.extract_video_features(frame_paths)
                if video_features is not None:
                    print(f"✓ Extracted video features shape: {video_features.shape}")
                else:
                    print("✗ Failed to extract video features")
        except Exception as e:
            print(f"✗ Error testing V-JEPA2: {e}")
            print("Make sure you have installed: pip install transformers torchcodec")
    else:
        print(f"Test directory {test_dir} not found")
        print("Available V-JEPA2 models:", list(VJEPA2_MODELS.keys()))
