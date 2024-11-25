import numpy as np
import cv2
import ffmpeg


class AdaptiveCompressor:
    """Applies adaptive compression based on saliency maps."""
    
    def __init__(self, high_quality_qp=18, low_quality_qp=35):
        """
        Initialize the adaptive compressor.
        
        Args:
            high_quality_qp: QP value for high-quality compression (salient regions)
            low_quality_qp: QP value for low-quality compression (non-salient regions)
        """
        self.high_quality_qp = high_quality_qp
        self.low_quality_qp = low_quality_qp
    
    def _create_qp_map(self, saliency_map):
        """Create QP map based on saliency values."""
        # Normalize saliency map to [0, 1]
        saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        
        # Create QP map by interpolating between high and low quality QP values
        qp_map = self.low_quality_qp - (self.low_quality_qp - self.high_quality_qp) * saliency_norm
        return qp_map.astype(np.uint8)
    
    def compress_frame(self, frame, saliency_map):
        """
        Compress a single frame using adaptive compression.
        
        Args:
            frame: Input frame (numpy array)
            saliency_map: Corresponding saliency map
            
        Returns:
            Compressed frame
        """
        qp_map = self._create_qp_map(saliency_map)
        
        # Split frame into blocks and compress each block
        height, width = frame.shape[:2]
        block_size = 16  # Standard macroblock size
        compressed_frame = np.zeros_like(frame)
        
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # Extract block
                block = frame[y:y+block_size, x:x+block_size]
                qp = int(np.mean(qp_map[y:y+block_size, x:x+block_size]))
                
                # Compress block using OpenCV's JPEG compression
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100 - qp]
                _, encoded = cv2.imencode('.jpg', block, encode_param)
                decoded = cv2.imdecode(encoded, 1)
                
                # Place back the compressed block
                h, w = decoded.shape[:2]
                compressed_frame[y:y+h, x:x+w] = decoded
        
        return compressed_frame
    
    def compress_video(self, input_path, output_path, saliency_maps):
        """
        Compress video using adaptive compression.
        
        Args:
            input_path: Path to input video
            output_path: Path to save compressed video
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
                
            # Compress frame
            compressed_frame = self.compress_frame(frame, saliency_maps[frame_idx])
            out.write(compressed_frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        # Use FFmpeg to finalize the video with proper encoding
        stream = ffmpeg.input(output_path)
        stream = ffmpeg.output(stream, output_path + '_final.mp4',
                             vcodec='libx264',
                             preset='medium',
                             crf=23)
        ffmpeg.run(stream, overwrite_output=True)
        
        # Replace original output with properly encoded version
        import os
        os.replace(output_path + '_final.mp4', output_path)
