# app.py
import streamlit as st
import logging.config
import os
from pathlib import Path
import time
import psutil
from utils import (
    ModelManager, 
    ImageProcessor, 
    SystemMonitor, 
    setup_logging,
    initialize_torch
)
from video_processor import VideoProcessor
from webrtc_handler import WebRTCHandler
from config import (
    APP_CONFIG, 
    LOGGING_CONFIG, 
    WEIGHTS_DIR,
    SOURCES_LIST,
    PERFORMANCE_CONFIG
)

# Initialize logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class UnderwaterDetectionApp:
    def __init__(self):
        self.setup_app()
        self.device = initialize_torch()
        self.system_monitor = SystemMonitor()
        self.model_manager = ModelManager()
        self.image_processor = ImageProcessor()
        self.webrtc_handler = WebRTCHandler()

    def setup_app(self):
        """Initialize Streamlit app configuration"""
        st.set_page_config(
            page_title=APP_CONFIG["title"],
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Set up session state
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'processing_active' not in st.session_state:
            st.session_state.processing_active = False

    def sidebar_controls(self):
        """Create sidebar controls"""
        with st.sidebar:
            st.title("Settings")
            
            # Model selection
            model_type = st.selectbox(
                "Select Model",
                list(APP_CONFIG["available_models"].keys()),
                index=0
            )
            
            # Confidence threshold
            confidence = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=APP_CONFIG["default_confidence"],
                step=0.05
            )
            
            # Input source selection
            source_type = st.selectbox(
                "Select Input Source",
                SOURCES_LIST
            )
            
            # System monitoring
            if st.checkbox("Show System Metrics"):
                self.system_monitor.display_metrics()
            
            return model_type, confidence, source_type

    def load_model(self, model_type):
        """Load the selected model"""
        try:
            if st.session_state.model is None:
                with st.spinner("Loading model..."):
                    model_path = WEIGHTS_DIR / APP_CONFIG["available_models"][model_type]
                    st.session_state.model = self.model_manager.load_model(
                        model_path, 
                        self.device
                    )
                    if st.session_state.model is None:
                        st.error("Failed to load model")
                        return False
                    st.success("Model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            st.error("Failed to load model. Please try again.")
            return False

    def handle_webcam(self, confidence):
        """Handle webcam input"""
        try:
            st.info("Initializing webcam stream...")
            webrtc_ctx = self.webrtc_handler.initialize_webrtc(
                confidence,
                st.session_state.model,
                PERFORMANCE_CONFIG
            )
            
            if webrtc_ctx and webrtc_ctx.state.playing:
                st.success("Stream started successfully")
            else:
                st.warning("Waiting for webcam connection...")
                
        except Exception as e:
            logger.error(f"Webcam error: {e}")
            st.error("Failed to initialize webcam stream")

    def handle_image(self, confidence):
        """Handle image input"""
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=APP_CONFIG["supported_image_types"]
        )
        
        if uploaded_file:
            self.image_processor.process_uploaded_image(
                uploaded_file,
                st.session_state.model,
                confidence
            )

    def handle_video(self, confidence):
        """Handle video input"""
        uploaded_file = st.file_uploader(
            "Upload Video",
            type=APP_CONFIG["supported_video_types"]
        )
        
        if uploaded_file:
            self.image_processor.process_uploaded_video(
                uploaded_file,
                st.session_state.model,
                confidence
            )

    def run(self):
        """Main application loop"""
        try:
            # Display app header
            st.title(APP_CONFIG["title"])
            st.markdown(APP_CONFIG["description"])
            
            # Get user inputs
            model_type, confidence, source_type = self.sidebar_controls()
            
            # Load model
            if not self.load_model(model_type):
                return
            
            # Process based on source type
            if source_type == "Webcam":
                self.handle_webcam(confidence)
            elif source_type == "Image":
                self.handle_image(confidence)
            elif source_type == "Video":
                self.handle_video(confidence)
            
            # Monitor system resources
            self.system_monitor.check_resources()
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error("An unexpected error occurred. Please try refreshing the page.")
            
        finally:
            # Cleanup
            if st.session_state.processing_active:
                self.system_monitor.cleanup()

def main():
    app = UnderwaterDetectionApp()
    app.run()

if __name__ == "__main__":
    main()