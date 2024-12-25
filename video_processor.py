# video_processor.py
import av
import cv2
import numpy as np
import torch
import time
import logging
from queue import Queue
from threading import Lock
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, confidence: float, model: YOLO, performance_config: dict):
        self.confidence = confidence
        self.model = model
        self.frame_count = 0
        self.last_process_time = time.time()
        self.performance_config = performance_config
        
        # Initialize frame buffer and processing queue
        self.frame_buffer = []
        self.processed_frames = Queue(maxsize=30)
        self.lock = Lock()
        
        # Initialize performance metrics
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        
        # Initialize batch processing
        self.batch_size = performance_config['batch_size']
        self.frame_skip = performance_config['frame_skip']
        self.target_resolution = performance_config['resolution']

    def update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
        self.fps_counter += 1

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame before model inference"""
        try:
            # Resize frame
            frame = cv2.resize(frame, self.target_resolution)
            
            # Normalize pixel values
            frame = frame.astype(np.float32) / 255.0
            
            return frame
        except Exception as e:
            logger.error(f"Frame preprocessing error: {e}")
            return frame

    def process_batch(self, batch: np.ndarray):
        """Process a batch of frames"""
        try:
            with torch.no_grad():
                results = self.model.predict(
                    batch,
                    conf=self.confidence,
                    device=next(self.model.parameters()).device
                )
            return results
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming video frames"""
        try:
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Skip frames based on configuration
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            current_time = time.time()
            if current_time - self.last_process_time < self.performance_config['processing_interval']:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Preprocess frame
            processed_frame = self.preprocess_frame(img)
            
            # Add to batch buffer
            with self.lock:
                self.frame_buffer.append(processed_frame)
                
                # Process batch when buffer is full
                if len(self.frame_buffer) >= self.batch_size:
                    batch = np.stack(self.frame_buffer)
                    results = self.process_batch(batch)
                    
                    if results is not None:
                        # Get the last processed frame
                        img = results[-1].plot()
                        self.frame_buffer.clear()
                        self.last_process_time = current_time
            
            # Update FPS
            self.update_fps()
            
            # Draw FPS on frame
            cv2.putText(
                img,
                f"FPS: {self.current_fps}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame

    def cleanup(self):
        """Cleanup resources"""
        with self.lock:
            self.frame_buffer.clear()
            while not self.processed_frames.empty():
                self.processed_frames.get()