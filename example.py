import os
from src.pipeline import HVSVideoPipeline


def main():
    # Initialize the pipeline with custom parameters
    pipeline = HVSVideoPipeline(
        high_quality_qp=20,      # Higher quality for salient regions
        low_quality_qp=35,       # Lower quality for non-salient regions
        sharpness_factor=1.3,    # Moderate sharpness enhancement
        contrast_factor=1.2      # Moderate contrast enhancement
    )
    
    # Example: Process a single video
    input_video = "path/to/your/video.mp4"
    output_dir = "path/to/output/directory"
    
    if os.path.exists(input_video):
        print(f"Processing video: {input_video}")
        result = pipeline.process_video(input_video, output_dir)
        
        print("\nProcessing complete!")
        print(f"Output saved to: {result['output_path']}")
        print("\nQuality Metrics:")
        for metric, values in result['quality_metrics'].items():
            print(f"\n{metric}:")
            for key, value in values.items():
                print(f"  {key}: {value:.4f}")
    
    # Example: Process all videos in a directory
    input_dir = "path/to/input/directory"
    if os.path.exists(input_dir):
        print(f"\nProcessing all videos in: {input_dir}")
        results = pipeline.process_batch(input_dir, output_dir)
        
        print("\nBatch processing complete!")
        for video_name, result in results.items():
            if 'error' in result:
                print(f"\nError processing {video_name}: {result['error']}")
            else:
                print(f"\nProcessed {video_name}")
                print(f"Output saved to: {result['output_path']}")


if __name__ == "__main__":
    main()
