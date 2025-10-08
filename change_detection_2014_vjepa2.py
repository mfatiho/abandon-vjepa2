#!/usr/bin/env python3
"""
V-JEPA2 Change Detection 2014 Dataset Utilities
Video-based change detection using V-JEPA2 self-supervised features
"""

import os
import contextlib
from pathlib import Path
from typing import Optional, List, Tuple, Union
import concurrent.futures

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# Import V-JEPA2 utilities
from extract_feats_vjepa2 import (
    VJEPA2_MODELS, VJEPA2_FEATURE_DIMS, DEFAULT_VJEPA2_MODEL,
    VJEPA2FeatureExtractor, load_vjepa2_model, get_vjepa2_feature_dim
)

# Constants
PATCH_SIZE = 16
IMAGE_SIZE = 224  # V-JEPA2 standard input size
TARGET_HEIGHT = 1088  # For Full-HD support

# Dataset configuration
LOCAL_DATA_DIR = "data/changedetection2014"
CHANGEDETECTION_URL = "http://www.changedetection.net"

# Change Detection 2014 categories and their video sequences
CD2014_CATEGORIES = {
    'baseline': ['highway', 'office', 'pedestrians', 'PETS2006'],
    'badWeather': ['blizzard', 'skating', 'snowFall', 'wetSnow'],
    'cameraJitter': ['badminton', 'boulevard', 'sidewalk', 'traffic'],
    'dynamicBackground': ['boats', 'canoe', 'fall', 'fountain01', 'fountain02', 'overpass'],
    'intermittentObjectMotion': ['abandonedBox', 'parking', 'sofa', 'streetLight', 'tramstop', 'winterDriveway'],
    'lowFramerate': ['port_0_17fps', 'tramCrossroad_1fps', 'tunnelExit_0_35fps', 'turnpike_0_5fps'],
    'nightVideos': ['bridgeEntry', 'busyBoulvard', 'fluidHighway', 'streetCornerAtNight', 'tramStation', 'winterStreet'],
    'PTZ': ['continuousPan', 'intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
    'shadow': ['backdoor', 'bungalows', 'busStation', 'copyMachine', 'cubicle', 'peopleInShade'],
    'thermal': ['corridor', 'diningRoom', 'lakeSide', 'library', 'park'],
    'turbulence': ['turbulence0', 'turbulence1', 'turbulence2', 'turbulence3']
}


def download_changedetection_dataset_info(data_dir: str, categories: List[str] = None):
    """Print download instructions for Change Detection 2014 dataset"""
    if categories is None:
        categories = ['baseline', 'dynamicBackground', 'shadow']
    
    print(f"Change Detection 2014 dataset setup required at: {data_dir}")
    print("Please manually download the Change Detection 2014 dataset from:")
    print("http://www.changedetection.net/")
    print(f"Extract it to: {data_dir}")
    print("Expected structure:")
    print(f"{data_dir}/")
    for category in categories:
        if category in CD2014_CATEGORIES:
            print(f"  {category}/")
            for video in CD2014_CATEGORIES[category]:
                print(f"    {video}/")
                print(f"      input/")
                print(f"      groundtruth/")
    
    return categories


def validate_dataset_structure(data_dir: str, categories: List[str]) -> List[str]:
    """Validate dataset structure and return available categories"""
    if not os.path.exists(data_dir):
        download_changedetection_dataset_info(data_dir, categories)
        raise FileNotFoundError(f"Please download the dataset manually to {data_dir}")
    
    # Validate categories
    available_categories = list(CD2014_CATEGORIES.keys())
    valid_categories = [cat for cat in categories if cat in available_categories]
    
    if len(valid_categories) != len(categories):
        missing = set(categories) - set(valid_categories)
        print(f"Warning: Unknown categories {missing}, using only: {valid_categories}")
    
    return valid_categories


def load_video_frames(input_dir: str, gt_dir: str, max_frames: int, 
                     smart_sampling: bool = True) -> Tuple[List[str], List[str]]:
    """Load and sample video frames"""
    # Get frame files
    input_files = sorted([f for f in os.listdir(input_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    gt_files = sorted([f for f in os.listdir(gt_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    # Smart frame sampling for longer videos
    if len(input_files) > max_frames:
        if smart_sampling:
            # Use more intelligent sampling - keep start, middle, and end frames
            start_frames = input_files[:max_frames//3]
            start_gt = gt_files[:max_frames//3]
            
            mid_start = len(input_files) // 2 - max_frames//6
            mid_end = len(input_files) // 2 + max_frames//6
            mid_frames = input_files[mid_start:mid_end]
            mid_gt = gt_files[mid_start:mid_end]
            
            end_frames = input_files[-max_frames//3:]
            end_gt = gt_files[-max_frames//3:]
            
            input_files = start_frames + mid_frames + end_frames
            gt_files = start_gt + mid_gt + end_gt
        else:
            # Simple uniform sampling
            step = len(input_files) // max_frames
            input_files = input_files[::step][:max_frames]
            gt_files = gt_files[::step][:max_frames]
    
    return input_files, gt_files


class ChangeDetection2014VJEPA2Loader:
    """V-JEPA2 based loader for Change Detection 2014 dataset"""
    
    def __init__(self, model_name: str = DEFAULT_VJEPA2_MODEL, device: str = "cuda", 
                 is_main_process: bool = True, max_frames_per_chunk: int = 16):
        """
        Initialize V-JEPA2 loader
        
        Args:
            model_name: V-JEPA2 model name
            device: Device to use
            is_main_process: Whether this is the main process
            max_frames_per_chunk: Maximum frames to process in one chunk
        """
        self.model_name = model_name
        self.device = device
        self.is_main_process = is_main_process
        self.max_frames_per_chunk = max_frames_per_chunk
        
        # Initialize V-JEPA2 feature extractor
        self.feature_extractor = VJEPA2FeatureExtractor(model_name, device, is_main_process)
    
    def load_videos(self, data_dir: str = LOCAL_DATA_DIR, categories: List[str] = None, 
                   max_frames_per_video: int = 300, validation_split: bool = True,
                   use_cache: bool = True, cache_dir: str = "cache") -> Union[List, Tuple[List, List]]:
        """
        Load Change Detection 2014 video sequences with V-JEPA2 features
        
        Args:
            data_dir: Dataset directory path
            categories: List of categories to load
            max_frames_per_video: Maximum frames per video
            validation_split: Whether to split into train/val
            use_cache: Whether to use caching
            cache_dir: Cache directory
        
        Returns:
            If validation_split: (train_sequences, val_sequences)
            Else: train_sequences
        """
        if self.is_main_process:
            print("Loading Change Detection 2014 dataset with V-JEPA2 features...")
        
        if categories is None:
            categories = ['baseline', 'dynamicBackground', 'shadow']
        
        # Validate dataset structure
        categories = validate_dataset_structure(data_dir, categories)
        
        if self.is_main_process:
            print(f"Using V-JEPA2 model: {self.model_name}")
            print(f"Using categories: {categories}")
            if validation_split:
                print("Using train/validation split (last video per category for validation)")
        
        # Check cache first
        cache_file = None
        if use_cache:
            cache_file = self._get_cache_file(cache_dir, categories, max_frames_per_video)
            cached_data = self._load_from_cache(cache_file)
            if cached_data is not None:
                return cached_data
        
        # Load data
        train_sequences = []
        val_sequences = []
        
        for category in categories:
            videos_in_category = CD2014_CATEGORIES[category]
            
            if self.is_main_process:
                print(f"Loading category {category} with {len(videos_in_category)} videos: {videos_in_category}")
            
            # Split videos: all but last for training, last for validation
            if validation_split and len(videos_in_category) > 1:
                train_videos = videos_in_category[:-1]
                val_videos = [videos_in_category[-1]]
            else:
                train_videos = videos_in_category
                val_videos = []
            
            # Process training videos
            for video_name in train_videos:
                sequences = self._load_single_video(data_dir, category, video_name, max_frames_per_video)
                train_sequences.extend(sequences)
            
            # Process validation videos
            for video_name in val_videos:
                sequences = self._load_single_video(data_dir, category, video_name, max_frames_per_video)
                val_sequences.extend(sequences)
        
        if self.is_main_process:
            print(f"Total training videos: {len(train_sequences)}")
            print(f"Total validation videos: {len(val_sequences)}")
        
        # Save to cache
        if use_cache and cache_file:
            cache_data = (train_sequences, val_sequences) if validation_split else train_sequences
            self._save_to_cache(cache_file, cache_data)
        
        return (train_sequences, val_sequences) if validation_split else train_sequences
    
    def _get_cache_file(self, cache_dir: str, categories: List[str], max_frames: int) -> str:
        """Generate cache file path"""
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = f"{'_'.join(sorted(categories))}_{max_frames}_{self.model_name}_vjepa2"
        return os.path.join(cache_dir, f"video_features_{cache_key}.pkl")
    
    def _load_from_cache(self, cache_file: str) -> Optional[Union[List, Tuple[List, List]]]:
        """Load data from cache with error handling"""
        if not os.path.exists(cache_file):
            return None
        
        if self.is_main_process:
            print(f"Loading from cache: {cache_file}")
        
        import pickle
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            if self.is_main_process:
                if isinstance(cached_data, tuple) and len(cached_data) == 2:
                    print(f"✓ Loaded {len(cached_data[0])} train and {len(cached_data[1])} val cached V-JEPA2 sequences")
                else:
                    print(f"✓ Loaded {len(cached_data)} cached V-JEPA2 sequences")
            return cached_data
        except (EOFError, pickle.UnpicklingError, FileNotFoundError) as e:
            if self.is_main_process:
                print(f"⚠️ Cache file corrupted or incomplete: {e}")
                print(f"Deleting corrupted cache: {cache_file}")
            try:
                os.remove(cache_file)
            except:
                pass
            if self.is_main_process:
                print("Regenerating cache from scratch...")
            return None
    
    def _save_to_cache(self, cache_file: str, cache_data):
        """Save data to cache with error handling"""
        if not self.is_main_process:
            return
        
        print(f"Saving V-JEPA2 features to cache: {cache_file}")
        import pickle
        try:
            # Save to temporary file first
            temp_cache_file = cache_file + ".tmp"
            with open(temp_cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Verify the file was written correctly
            with open(temp_cache_file, 'rb') as f:
                test_load = pickle.load(f)
            
            # If verification passes, move to final location
            import shutil
            shutil.move(temp_cache_file, cache_file)
            
            print("✓ V-JEPA2 cache saved and verified successfully")
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
            # Clean up temporary file if it exists
            try:
                if os.path.exists(temp_cache_file):
                    os.remove(temp_cache_file)
            except:
                pass
    
    def _load_single_video(self, data_dir: str, category: str, video_name: str, 
                          max_frames: int) -> List:
        """Load a single video sequence with V-JEPA2 features"""
        video_dir = os.path.join(data_dir, category, video_name)
        input_dir = os.path.join(video_dir, 'input')
        gt_dir = os.path.join(video_dir, 'groundtruth')
        
        if not (os.path.exists(input_dir) and os.path.exists(gt_dir)):
            if self.is_main_process:
                print(f"Warning: Video {category}/{video_name} not found, skipping...")
            return []
        
        if self.is_main_process:
            print(f"Processing video with V-JEPA2: {category}/{video_name}")
        
        # Load frame files with smart sampling
        input_files, gt_files = load_video_frames(input_dir, gt_dir, max_frames, smart_sampling=True)
        
        # Extract V-JEPA2 features
        video_features, video_labels = self.feature_extractor.extract_video_sequential(
            input_dir, gt_dir, input_files, gt_files
        )
        
        if len(video_features) > 0:
            if self.is_main_process:
                print(f"Processed {len(video_features)} frames with V-JEPA2 from {category}/{video_name}")
            return [(video_features, video_labels)]
        
        return []


# Convenience functions
def load_changedetection_videos_vjepa2(model_name: str = DEFAULT_VJEPA2_MODEL, device: str = "cuda", 
                                      data_dir: str = LOCAL_DATA_DIR, categories: List[str] = None, 
                                      max_frames_per_video: int = 300, validation_split: bool = True,
                                      use_cache: bool = True, cache_dir: str = "cache",
                                      is_main_process: bool = True) -> Union[List, Tuple[List, List]]:
    """
    Convenience function to load Change Detection 2014 videos with V-JEPA2 features
    
    Args:
        model_name: V-JEPA2 model name
        device: Device to use
        data_dir: Dataset directory path
        categories: List of categories to load
        max_frames_per_video: Maximum frames per video
        validation_split: Whether to split into train/val
        use_cache: Whether to use caching
        cache_dir: Cache directory
        is_main_process: Whether this is the main process
    
    Returns:
        If validation_split: (train_sequences, val_sequences)
        Else: train_sequences
    """
    loader = ChangeDetection2014VJEPA2Loader(model_name, device, is_main_process)
    return loader.load_videos(data_dir, categories, max_frames_per_video, validation_split, 
                             use_cache, cache_dir)


def extract_vjepa2_features_from_video_dir(video_dir: str, model_name: str = DEFAULT_VJEPA2_MODEL,
                                          device: str = "cuda", max_frames: int = 100) -> Tuple[np.ndarray, List[str]]:
    """
    Extract V-JEPA2 features from a directory of video frames
    
    Args:
        video_dir: Directory containing video frames
        model_name: V-JEPA2 model name
        device: Device to use
        max_frames: Maximum frames to process
        
    Returns:
        Tuple of (features_array, frame_filenames)
    """
    extractor = load_vjepa2_model(model_name, device)
    
    # Get frame files
    frame_files = sorted([f for f in os.listdir(video_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    if len(frame_files) > max_frames:
        # Sample frames uniformly
        indices = np.linspace(0, len(frame_files)-1, max_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    
    # Extract features
    frame_paths = [os.path.join(video_dir, f) for f in frame_files]
    features = extractor.extract_video_features(frame_paths, max_frames=len(frame_files))
    
    return features, frame_files


if __name__ == "__main__":
    # Example usage
    print("V-JEPA2 Change Detection 2014 Dataset Utilities")
    print("Available V-JEPA2 models:", list(VJEPA2_MODELS.keys()))
    print("Available categories:", list(CD2014_CATEGORIES.keys()))
    
    # Test dataset structure validation
    try:
        categories = validate_dataset_structure(LOCAL_DATA_DIR, ['baseline', 'shadow'])
        print(f"Valid categories: {categories}")
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
    
    # Test V-JEPA2 loader
    try:
        loader = ChangeDetection2014VJEPA2Loader()
        print("V-JEPA2 loader initialized successfully")
    except Exception as e:
        print(f"Error initializing V-JEPA2 loader: {e}")
