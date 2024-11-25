import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


class VideoEnhancer:
    """Enhances video quality using HVS-inspired techniques."""
    
    def __init__(self, sharpness_factor=1.5, contrast_factor=1.2):
        """
        Initialize the video enhancer.
        
        Args:
            sharpness_factor: Factor to control sharpness enhancement
            contrast_factor: Factor to control contrast enhancement
        """
        self.sharpness_factor = sharpness_factor
        self.contrast_factor = contrast_factor
    
    def _enhance_details(self, frame, saliency_map):
        """Enhance fine details in salient regions."""
        # Convert to LAB color space for better perceptual processing
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        
        # Apply unsharp masking weighted by saliency
        blurred = gaussian_filter(l_channel, sigma=2)
        detail_mask = l_channel.astype(float) - blurred
        
        # Weight detail enhancement by saliency
        enhanced_details = l_channel + detail_mask * self.sharpness_factor * saliency_map
        
        # Clip values to valid range
        enhanced_details = np.clip(enhanced_details, 0, 255).astype(np.uint8)
        
        # Replace L channel
        lab[:,:,0] = enhanced_details
        
        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _enhance_contrast(self, frame, saliency_map):
        """Enhance contrast adaptively based on saliency."""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0].astype(float)
        
        # Calculate adaptive contrast enhancement
        mean_l = np.mean(l_channel)
        contrast_enhancement = (l_channel - mean_l) * self.contrast_factor * saliency_map
        
        # Apply enhancement
        enhanced_l = l_channel + contrast_enhancement
        
        # Clip values
        lab[:,:,0] = np.clip(enhanced_l, 0, 255).astype(np.uint8)
        
        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def enhance_frame(self, frame, saliency_map):
        """
        Enhance a single frame using HVS-based techniques.
        
        Args:
            frame: Input frame
            saliency_map: Corresponding saliency map
            
        Returns:
            Enhanced frame
        """
        # Normalize saliency map
        saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        
        # Apply enhancements
        enhanced = self._enhance_details(frame, saliency_norm)
        enhanced = self._enhance_contrast(enhanced, saliency_norm)
        
        return enhanced
    
    def enhance_video(self, input_path, output_path, saliency_maps):
        """
        Enhance video using HVS-based techniques.
        
        Args:
            input_path: Path to input video
            output_path: Path to save enhanced video
            saliency_maps: List of saliency maps for each frame
        """
        # Read input video
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Enhance frame
            enhanced_frame = self.enhance_frame(frame, saliency_maps[frame_idx])
            out.write(enhanced_frame)
            frame_idx += 1
        
        cap.release()
        out.release()
