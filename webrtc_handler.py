# webrtc_handler.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import logging
from typing import Optional
from video_processor import VideoProcessor
from config import WEBRTC_CONFIG
import time
from twilio.rest import Client
import os

logger = logging.getLogger(__name__)

class WebRTCHandler:
    def __init__(self):
        self.ice_servers = self.get_ice_servers()
        self.retry_count = 0
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    def get_ice_servers(self) -> list:
        """Get ICE servers configuration"""
        try:
            # Try to get Twilio credentials
            account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
            auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
            
            if account_sid and auth_token:
                # Use Twilio's TURN server
                client = Client(account_sid, auth_token)
                token = client.tokens.create()
                return token.ice_servers
            else:
                # Fallback to free STUN server
                logger.warning("Using fallback STUN server")
                return [{"urls": ["stun:stun.l.google.com:19302"]}]
                
        except Exception as e:
            logger.error(f"Error getting ICE servers: {e}")
            return [{"urls": ["stun:stun.l.google.com:19302"]}]

    def initialize_webrtc(self, confidence: float, model, performance_config: dict) -> Optional[bool]:
        """Initialize WebRTC connection with retry mechanism"""
        while self.retry_count < self.max_retries:
            try:
                rtc_configuration = RTCConfiguration(
                    {"iceServers": self.ice_servers}
                )
                
                webrtc_ctx = webrtc_streamer(
                    key="underwater-detection",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=rtc_configuration,
                    video_processor_factory=lambda: VideoProcessor(
                        confidence,
                        model,
                        performance_config
                    ),
                    media_stream_constraints=WEBRTC_CONFIG["media_stream_constraints"],
                    async_processing=True,
                    video_html_attrs={
                        "style": {"width": "100%", "height": "100%"},
                        "controls": False,
                        "autoPlay": True,
                    },
                )
                
                if webrtc_ctx.state.playing:
                    logger.info("WebRTC connection established successfully")
                    return webrtc_ctx
                    
            except Exception as e:
                self.retry_count += 1
                logger.warning(f"WebRTC connection attempt {self.retry_count} failed: {e}")
                time.sleep(self.retry_delay)
                
        logger.error("Failed to establish WebRTC connection after maximum retries")
        return None

    def cleanup(self):
        """Cleanup WebRTC resources"""
        try:
            # Additional cleanup if needed
            pass
        except Exception as e:
            logger.error(f"Error during WebRTC cleanup: {e}")