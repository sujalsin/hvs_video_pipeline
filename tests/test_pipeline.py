import os
import sys
import cv2
import numpy as np
import pytest

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.saliency.model import VideoSaliencyPredictor
from src.compression.adaptive_compressor import AdaptiveCompressor
from src.enhancement.enhancer import VideoEnhancer
from src.quality.assessor import QualityAssessor
from src.pipeline import HVSVideoPipeline


def create_test_video(output_path, num_frames=30):
    """Create a simple test video with moving shapes."""
    width, height = 320, 240
    fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        # Create a frame with moving shapes
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add moving circle
        cx = int(width/2 + width/4 * np.sin(i/10))
        cy = int(height/2)
        cv2.circle(frame, (cx, cy), 30, (0, 0, 255), -1)
        
        # Add moving rectangle
        rx = int(width/2 - width/4 * np.cos(i/10))
        ry = int(height/2)
        cv2.rectangle(frame, (rx-20, ry-20), (rx+20, ry+20), (0, 255, 0), -1)
        
        out.write(frame)
    
    out.release()
    return output_path


def test_saliency_detector():
    """Test the saliency detection module."""
    # Create a simple test frame
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (160, 120), 30, (0, 0, 255), -1)
    
    # Test saliency prediction
    predictor = VideoSaliencyPredictor()
    saliency_map = predictor.predict_frame(frame)
    
    assert saliency_map is not None
    assert saliency_map.shape == (240, 320)
    assert np.min(saliency_map) >= 0 and np.max(saliency_map) <= 1


def test_adaptive_compressor():
    """Test the adaptive compression module."""
    # Create test frame and saliency map
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (160, 120), 30, (0, 0, 255), -1)
    saliency_map = np.zeros((240, 320))
    saliency_map[90:150, 130:190] = 1
    
    # Test compression
    compressor = AdaptiveCompressor()
    compressed_frame = compressor.compress_frame(frame, saliency_map)
    
    assert compressed_frame is not None
    assert compressed_frame.shape == frame.shape


def test_video_enhancer():
    """Test the video enhancement module."""
    # Create test frame and saliency map
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (160, 120), 30, (0, 0, 255), -1)
    saliency_map = np.zeros((240, 320))
    saliency_map[90:150, 130:190] = 1
    
    # Test enhancement
    enhancer = VideoEnhancer()
    enhanced_frame = enhancer.enhance_frame(frame, saliency_map)
    
    assert enhanced_frame is not None
    assert enhanced_frame.shape == frame.shape


def test_quality_assessor():
    """Test the quality assessment module."""
    # Create test frames
    frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(frame1, (160, 120), 30, (0, 0, 255), -1)
    frame2 = frame1.copy()
    cv2.GaussianBlur(frame2, (5, 5), 0, frame2)
    
    # Test quality assessment
    assessor = QualityAssessor()
    metrics = assessor.assess_frame_quality(frame1, frame2)
    
    assert metrics is not None
    assert 'ssim' in metrics
    assert 'sharpness_original' in metrics
    assert 'sharpness_processed' in metrics


def test_full_pipeline():
    """Test the complete pipeline."""
    # Create test video
    test_video_path = os.path.join('tests', 'data', 'test_video.mp4')
    output_dir = os.path.join('tests', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    create_test_video(test_video_path)
    
    # Test pipeline
    pipeline = HVSVideoPipeline()
    result = pipeline.process_video(test_video_path, output_dir)
    
    assert result is not None
    assert 'output_path' in result
    assert 'quality_metrics' in result
    assert os.path.exists(result['output_path'])


if __name__ == '__main__':
    # Run all tests
    print("Running tests...")
    
    print("\nTesting saliency detector...")
    test_saliency_detector()
    print("✓ Saliency detector test passed")
    
    print("\nTesting adaptive compressor...")
    test_adaptive_compressor()
    print("✓ Adaptive compressor test passed")
    
    print("\nTesting video enhancer...")
    test_video_enhancer()
    print("✓ Video enhancer test passed")
    
    print("\nTesting quality assessor...")
    test_quality_assessor()
    print("✓ Quality assessor test passed")
    
    print("\nTesting full pipeline...")
    test_full_pipeline()
    print("✓ Full pipeline test passed")
    
    print("\nAll tests passed successfully!")
