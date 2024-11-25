import os
from .saliency.model import VideoSaliencyPredictor
from .compression.adaptive_compressor import AdaptiveCompressor
from .enhancement.enhancer import VideoEnhancer
from .quality.assessor import QualityAssessor


class HVSVideoPipeline:
    """Main pipeline class that integrates all HVS-based video processing components."""
    
    def __init__(self,
                 high_quality_qp=18,
                 low_quality_qp=35,
                 sharpness_factor=1.5,
                 contrast_factor=1.2):
        """
        Initialize the HVS video pipeline.
        
        Args:
            high_quality_qp: QP value for high-quality compression
            low_quality_qp: QP value for low-quality compression
            sharpness_factor: Factor for sharpness enhancement
            contrast_factor: Factor for contrast enhancement
        """
        self.saliency_predictor = VideoSaliencyPredictor()
        self.compressor = AdaptiveCompressor(high_quality_qp, low_quality_qp)
        self.enhancer = VideoEnhancer(sharpness_factor, contrast_factor)
        self.quality_assessor = QualityAssessor()
        
    def process_video(self, input_path, output_dir, enhance_first=True):
        """
        Process a video through the HVS-inspired pipeline.
        
        Args:
            input_path: Path to input video
            output_dir: Directory to save outputs
            enhance_first: Whether to enhance before compression
            
        Returns:
            Dictionary containing quality metrics and output paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate base output paths
        basename = os.path.splitext(os.path.basename(input_path))[0]
        enhanced_path = os.path.join(output_dir, f"{basename}_enhanced.mp4")
        compressed_path = os.path.join(output_dir, f"{basename}_compressed.mp4")
        final_path = os.path.join(output_dir, f"{basename}_final.mp4")
        
        # Extract saliency maps
        print("Generating saliency maps...")
        import cv2
        cap = cv2.VideoCapture(input_path)
        saliency_maps = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            saliency_map = self.saliency_predictor.predict_frame(frame)
            saliency_maps.append(saliency_map)
        cap.release()
        
        # Process video
        if enhance_first:
            print("Enhancing video...")
            self.enhancer.enhance_video(input_path, enhanced_path, saliency_maps)
            
            print("Compressing enhanced video...")
            self.compressor.compress_video(enhanced_path, final_path, saliency_maps)
        else:
            print("Compressing video...")
            self.compressor.compress_video(input_path, compressed_path, saliency_maps)
            
            print("Enhancing compressed video...")
            self.enhancer.enhance_video(compressed_path, final_path, saliency_maps)
        
        # Assess quality
        print("Assessing video quality...")
        quality_metrics = self.quality_assessor.assess_video_quality(
            input_path, final_path, saliency_maps
        )
        
        # Clean up intermediate files
        if os.path.exists(enhanced_path):
            os.remove(enhanced_path)
        if os.path.exists(compressed_path):
            os.remove(compressed_path)
        
        return {
            'output_path': final_path,
            'quality_metrics': quality_metrics
        }
    
    def process_batch(self, input_dir, output_dir, enhance_first=True):
        """
        Process multiple videos in a directory.
        
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory to save outputs
            enhance_first: Whether to enhance before compression
            
        Returns:
            Dictionary containing results for each video
        """
        results = {}
        for filename in os.listdir(input_dir):
            if filename.endswith(('.mp4', '.avi', '.mov')):
                input_path = os.path.join(input_dir, filename)
                print(f"\nProcessing {filename}...")
                try:
                    results[filename] = self.process_video(
                        input_path, output_dir, enhance_first
                    )
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    results[filename] = {'error': str(e)}
        
        return results
