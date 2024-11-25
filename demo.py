import os
import cv2
from src.pipeline import HVSVideoPipeline


def main():
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a sample video with moving shapes
    print("Creating sample video...")
    width, height = 640, 480
    fps = 30
    duration = 5  # seconds
    
    sample_path = os.path.join(output_dir, "sample.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(sample_path, fourcc, fps, (width, height))
    
    for i in range(duration * fps):
        # Create frame with moving shapes
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add moving circle
        cx = int(width/2 + width/4 * np.sin(i/30))
        cy = int(height/2)
        cv2.circle(frame, (cx, cy), 50, (0, 0, 255), -1)
        
        # Add moving rectangle
        rx = int(width/2 - width/4 * np.cos(i/30))
        ry = int(height/2)
        cv2.rectangle(frame, (rx-40, ry-40), (rx+40, ry+40), (0, 255, 0), -1)
        
        # Add some text
        cv2.putText(frame, "HVS Video Pipeline Demo", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Sample video created at: {sample_path}")
    
    # Initialize the pipeline
    print("\nInitializing HVS video pipeline...")
    pipeline = HVSVideoPipeline(
        high_quality_qp=20,      # Higher quality for salient regions
        low_quality_qp=35,       # Lower quality for non-salient regions
        sharpness_factor=1.3,    # Moderate sharpness enhancement
        contrast_factor=1.2      # Moderate contrast enhancement
    )
    
    # Process the video
    print("\nProcessing video...")
    result = pipeline.process_video(sample_path, output_dir)
    
    print("\nProcessing complete!")
    print(f"Output saved to: {result['output_path']}")
    print("\nQuality Metrics:")
    for metric, values in result['quality_metrics'].items():
        print(f"\n{metric}:")
        for key, value in values.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    import numpy as np
    main()
