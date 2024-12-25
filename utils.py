# utils.py
import streamlit as st
import cv2
import torch
import numpy as np
import logging
import psutil
import GPUtil
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import os
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def initialize_torch():
    """Initialize PyTorch settings"""
    if torch.cuda.is_available():
        # Optimize CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.7)
        
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SystemMonitor:
    """Monitor system resources"""
    
    def __init__(self):
        self.start_time = time.time()
    
    def get_system_metrics(self) -> dict:
        """Get current system metrics"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
        }
        
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            metrics.update({
                'gpu_load': gpu.load * 100,
                'gpu_memory_percent': gpu.memoryUtil * 100
            })
            
        return metrics
    
    def check_resources(self):
        """Check if system resources are within acceptable limits"""
        metrics = self.get_system_metrics()
        
        if metrics['memory_percent'] > 90:
            logger.warning("High memory usage detected")
            torch.cuda.empty_cache()
            
        if metrics['cpu_percent'] > 80:
            logger.warning("High CPU usage detected")
    
    def display_metrics(self):
        """Display system metrics in Streamlit"""
        metrics = self.get_system_metrics()
        
        st.sidebar.subheader("System Metrics")
        st.sidebar.progress(metrics['cpu_percent'] / 100)
        st.sidebar.text(f"CPU Usage: {metrics['cpu_percent']}%")
        st.sidebar.progress(metrics['memory_percent'] / 100)
        st.sidebar.text(f"Memory Usage: {metrics['memory_percent']}%")
        
        if 'gpu_load' in metrics:
            st.sidebar.progress(metrics['gpu_load'] / 100)
            st.sidebar.text(f"GPU Usage: {metrics['gpu_load']:.1f}%")
    
    def cleanup(self):
        """Cleanup system resources"""
        torch.cuda.empty_cache()

class ModelManager:
    """Handle model loading and management"""
    
    @staticmethod
    @st.cache_resource
    def load_model(model_path: Path, device: torch.device) -> Optional[YOLO]:
        try:
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            model = YOLO(str(model_path))
            model.to(device)
            
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

class ImageProcessor:
    """Handle image processing operations"""
    
    def process_uploaded_image(self, uploaded_file, model, confidence):
        """Process uploaded image file"""
        try:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image")
            
            results = model.predict(image_np, conf=confidence)
            processed_image = results[0].plot()
            
            with col2:
                st.image(processed_image, caption="Processed Image")
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            st.error("Failed to process image")
    
    def process_uploaded_video(self, uploaded_file, model, confidence):
        """Process uploaded video file"""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(temp_file.name)
            stframe = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                results = model.predict(frame, conf=confidence)
                processed_frame = results[0].plot()
                
                stframe.image(processed_frame, channels="BGR")
                
            cap.release()
            os.unlink(temp_file.name)
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            st.error("Failed to process video")