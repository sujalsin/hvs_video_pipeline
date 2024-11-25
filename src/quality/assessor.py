import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy


class QualityAssessor:
    """Assesses video quality using HVS-aligned metrics."""
    
    def __init__(self):
        """Initialize the quality assessor."""
        pass
    
    def _calculate_ssim(self, frame1, frame2):
        """Calculate SSIM between two frames."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate SSIM
        score, _ = ssim(gray1, gray2, full=True)
        return score
    
    def _calculate_perceptual_sharpness(self, frame):
        """Calculate perceptual sharpness using gradient magnitude."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(gx**2 + gy**2)
        
        return np.mean(magnitude)
    
    def _calculate_color_naturalness(self, frame):
        """Calculate color naturalness using color distribution."""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram of hue channel
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist = hist.flatten() / hist.sum()
        
        # Calculate entropy of hue distribution
        color_entropy = entropy(hist)
        
        return color_entropy
    
    def assess_frame_quality(self, original_frame, processed_frame, saliency_map=None):
        """
        Assess the quality of a processed frame compared to the original.
        
        Args:
            original_frame: Original input frame
            processed_frame: Processed (enhanced/compressed) frame
            saliency_map: Optional saliency map for weighted assessment
            
        Returns:
            Dictionary containing quality metrics
        """
        metrics = {}
        
        # Calculate SSIM
        metrics['ssim'] = self._calculate_ssim(original_frame, processed_frame)
        
        # Calculate sharpness
        metrics['sharpness_original'] = self._calculate_perceptual_sharpness(original_frame)
        metrics['sharpness_processed'] = self._calculate_perceptual_sharpness(processed_frame)
        
        # Calculate color naturalness
        metrics['color_naturalness'] = self._calculate_color_naturalness(processed_frame)
        
        if saliency_map is not None:
            # Weight metrics by saliency
            saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            metrics['weighted_ssim'] = metrics['ssim'] * np.mean(saliency_norm)
        
        return metrics
    
    def assess_video_quality(self, original_path, processed_path, saliency_maps=None):
        """
        Assess the quality of a processed video compared to the original.
        
        Args:
            original_path: Path to original video
            processed_path: Path to processed video
            saliency_maps: Optional list of saliency maps
            
        Returns:
            Dictionary containing aggregated quality metrics
        """
        cap_original = cv2.VideoCapture(original_path)
        cap_processed = cv2.VideoCapture(processed_path)
        
        frame_metrics = []
        frame_idx = 0
        
        while cap_original.isOpened() and cap_processed.isOpened():
            ret1, frame1 = cap_original.read()
            ret2, frame2 = cap_processed.read()
            
            if not ret1 or not ret2:
                break
            
            saliency = saliency_maps[frame_idx] if saliency_maps is not None else None
            metrics = self.assess_frame_quality(frame1, frame2, saliency)
            frame_metrics.append(metrics)
            frame_idx += 1
        
        cap_original.release()
        cap_processed.release()
        
        # Aggregate metrics
        aggregated_metrics = {}
        for key in frame_metrics[0].keys():
            values = [m[key] for m in frame_metrics]
            aggregated_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return aggregated_metrics
