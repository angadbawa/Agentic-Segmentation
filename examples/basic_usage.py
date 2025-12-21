"""
Basic usage examples for the Agentic Segmentation Pipeline

This script demonstrates how to use the zero-context agentic pipeline
for object detection and segmentation with Qwen 2.5/3 MLLM integration.
"""

import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.optimized_pipeline import OptimizedAgenticPipeline, OptimizedConfig

def basic_example():
    """Basic example of using the pipeline."""
    print("=== Basic Usage Example ===")
    
    pipeline = OptimizedAgenticPipeline(OptimizedConfig())
    
    image_path = "sample_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Please provide a valid image path. Current path: {image_path}")
        return
    
    print("Processing image...")
    result = pipeline.process_image(image_path)
    print(f"Detected {len(result.detected_objects)} objects:")
    for i, obj in enumerate(result.detected_objects):
        print(f"  {i+1}. {obj['label']} (confidence: {obj['confidence']:.3f})")
    
    print(f"Processing time: {result.processing_time:.2f} seconds")
    print(f"Model info: {result.model_info}")

def advanced_example():
    """Advanced example with custom configuration."""
    print("\n=== Advanced Usage Example ===")
    
    config = OptimizedConfig(
        mode="quality",
        confidence_threshold=0.4,
        max_objects=30
    )
    
    pipeline = OptimizedAgenticPipeline(config)
    
    image_path = "sample_image.jpg"  # Replace with actual image path
    
    if not os.path.exists(image_path):
        print(f"Please provide a valid image path. Current path: {image_path}")
        return
    
    modes = ["automatic", "guided"]
    
    for mode in modes:
        print(f"\nProcessing in {mode} mode...")
        result = pipeline.process_image(image_path, mode=mode)
        
        print(f"  Detected {len(result.detected_objects)} objects")
        print(f"  Processing time: {result.processing_time:.2f}s")
        
        # Show top 3 objects
        top_objects = sorted(result.detected_objects, 
                           key=lambda x: x['confidence'], reverse=True)[:3]
        for obj in top_objects:
            print(f"    - {obj['label']}: {obj['confidence']:.3f}")

def batch_processing_example():
    """Example of batch processing multiple images."""
    print("\n=== Batch Processing Example ===")
    
    # Use fast configuration for batch processing
    config = OptimizedConfig(mode="fast")
    pipeline = OptimizedAgenticPipeline(config)
    
    # List of image paths (replace with actual paths)
    image_paths = [
        "image1.jpg",
        "image2.jpg", 
        "image3.jpg"
    ]
    
    # Filter existing images
    existing_images = [path for path in image_paths if os.path.exists(path)]
    
    if not existing_images:
        print("Please provide valid image paths for batch processing")
        return
    
    print(f"Processing {len(existing_images)} images...")
    
    # Process batch
    results = pipeline.process_batch(existing_images, mode="automatic", parallel=True)
    
    # Display results
    for i, result in enumerate(results):
        print(f"Image {i+1}: {len(result.detected_objects)} objects detected")
        if result.detected_objects:
            best_object = max(result.detected_objects, key=lambda x: x['confidence'])
            print(f"  Best detection: {best_object['label']} ({best_object['confidence']:.3f})")

def database_management_example():
    """Example of managing the object database."""
    print("\n=== Database Management Example ===")
    
    pipeline = OptimizedAgenticPipeline(OptimizedConfig())
    
    image_path = "sample_image.jpg"  # Replace with actual image path
    
    if os.path.exists(image_path):
        pipeline.add_object_to_database(
            image=image_path,
            object_name="custom_object",
            description="A custom object for recognition",
            bbox=[100, 100, 200, 200]  # Example bounding box
        )
        
        print("Added custom object to database")
        
        pipeline.save_database("my_object_database.json")
        print("Saved database to file")
        
        pipeline.load_database("my_object_database.json")
        print("Loaded database from file")

def visualization_example():
    """Example of visualizing results."""
    print("\n=== Visualization Example ===")
    
    pipeline = OptimizedAgenticPipeline(OptimizedConfig())
    
    image_path = "sample_image.jpg"  # Replace with actual image path
    
    if not os.path.exists(image_path):
        print(f"Please provide a valid image path. Current path: {image_path}")
        return
    
    # Process image
    result = pipeline.process_image(image_path)
    
    # Visualize results
    image = Image.open(image_path)
    visualized = pipeline.visualize_results(image, result, alpha=0.6)
    
    # Save visualized result
    output_path = "visualized_result.jpg"
    visualized.save(output_path)
    print(f"Saved visualized result to {output_path}")
    
    # Display using matplotlib
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        ax2.imshow(visualized)
        ax2.set_title("Detection Results")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig("comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Saved comparison image to comparison.png")
        
    except ImportError:
        print("Matplotlib not available for display")

def performance_analysis_example():
    """Example of analyzing pipeline performance."""
    print("\n=== Performance Analysis Example ===")
    
    pipeline = OptimizedAgenticPipeline(OptimizedConfig())
    
    performance = pipeline.get_performance_summary()
    
    if performance:
        print("Performance Summary:")
        for metric, stats in performance.items():
            if isinstance(stats, dict):
                print(f"  {metric}:")
                for key, value in stats.items():
                    print(f"    {key}: {value}")
            else:
                print(f"  {metric}: {stats}")
    else:
        print("No performance data available yet")
    
    # Optimize configuration
    optimized_config = pipeline.optimize_configuration()
    print(f"\nOptimized configuration: {optimized_config}")

def main():
    """Run all examples."""
    print("Agentic Segmentation Pipeline Examples")
    print("=" * 50)
    
    # Run examples
    basic_example()
    advanced_example()
    batch_processing_example()
    database_management_example()
    visualization_example()
    performance_analysis_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")

if __name__ == "__main__":
    main()
