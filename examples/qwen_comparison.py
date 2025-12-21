"""
Comparison between Qwen-enhanced and standard pipeline

This script demonstrates the differences between using Qwen 2.5/3 MLLM
and the standard CLIP-based approach for object detection and segmentation.
"""

import os
import sys
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.optimized_pipeline import OptimizedAgenticPipeline, OptimizedConfig

def compare_pipelines():
    """Compare Qwen-enhanced vs standard pipeline."""
    print("=== Pipeline Comparison: Qwen vs Standard ===")
    
    # Load test image
    image_path = "sample_image.jpg"  # Replace with actual image path
    
    if not os.path.exists(image_path):
        print(f"Please provide a valid image path. Current path: {image_path}")
        return
    
    image = Image.open(image_path)
    print(f"Processing image: {image.size}")
    
    # Test configurations
    configs = {
        "Qwen Enhanced": PipelineConfig(
            use_qwen=True,
            qwen_model="Qwen/Qwen2.5-VL-7B-Instruct",
            sam_model="vit_l",
            confidence_threshold=0.4,
            detail_level="comprehensive"
        ),
        "Standard CLIP": PipelineConfig(
            use_qwen=False,
            sam_model="vit_l",
            confidence_threshold=0.4
        )
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n--- Testing {config_name} ---")
        
        # Initialize pipeline
        pipeline = AgenticSegmentationPipeline(config)
        
        # Process image
        start_time = time.time()
        result = pipeline.process_image(image, mode="guided")
        processing_time = time.time() - start_time
        
        # Store results
        results[config_name] = {
            'result': result,
            'processing_time': processing_time,
            'config': config
        }
        
        # Display results
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Objects detected: {len(result.detected_objects)}")
        print(f"Model info: {result.model_info}")
        
        # Show top detections
        if result.detected_objects:
            top_objects = sorted(result.detected_objects, 
                               key=lambda x: x['confidence'], reverse=True)[:5]
            print("Top detections:")
            for i, obj in enumerate(top_objects):
                print(f"  {i+1}. {obj['label']}: {obj['confidence']:.3f}")
    
    # Compare results
    print("\n=== Comparison Summary ===")
    
    qwen_result = results["Qwen Enhanced"]['result']
    standard_result = results["Standard CLIP"]['result']
    
    print(f"Qwen Enhanced:")
    print(f"  - Objects detected: {len(qwen_result.detected_objects)}")
    print(f"  - Processing time: {results['Qwen Enhanced']['processing_time']:.2f}s")
    print(f"  - Avg confidence: {np.mean(qwen_result.confidence_scores):.3f}")
    
    print(f"Standard CLIP:")
    print(f"  - Objects detected: {len(standard_result.detected_objects)}")
    print(f"  - Processing time: {results['Standard CLIP']['processing_time']:.2f}s")
    print(f"  - Avg confidence: {np.mean(standard_result.confidence_scores):.3f}")
    
    # Calculate improvements
    if len(standard_result.detected_objects) > 0:
        detection_improvement = (len(qwen_result.detected_objects) - len(standard_result.detected_objects)) / len(standard_result.detected_objects) * 100
        print(f"\nDetection improvement: {detection_improvement:+.1f}%")
    
    if len(standard_result.confidence_scores) > 0 and len(qwen_result.confidence_scores) > 0:
        confidence_improvement = (np.mean(qwen_result.confidence_scores) - np.mean(standard_result.confidence_scores)) / np.mean(standard_result.confidence_scores) * 100
        print(f"Confidence improvement: {confidence_improvement:+.1f}%")
    
    return results

def detailed_analysis():
    """Detailed analysis of Qwen vs standard approach."""
    print("\n=== Detailed Analysis ===")
    
    image_path = "sample_image.jpg"  # Replace with actual image path
    
    if not os.path.exists(image_path):
        print(f"Please provide a valid image path. Current path: {image_path}")
        return
    
    detail_levels = ["basic", "detailed", "comprehensive"]
    
    for detail_level in detail_levels:
        print(f"\n--- Testing detail level: {detail_level} ---")
        
        # Qwen with specific detail level
        qwen_config = PipelineConfig(
            use_qwen=True,
            detail_level=detail_level,
            confidence_threshold=0.3
        )
        
        qwen_pipeline = AgenticSegmentationPipeline(qwen_config)
        qwen_result = qwen_pipeline.process_image(image_path, mode="guided")
        
        print(f"Qwen ({detail_level}):")
        print(f"  - Objects: {len(qwen_result.detected_objects)}")
        print(f"  - Avg confidence: {np.mean(qwen_result.confidence_scores):.3f}")
        
        # Show object labels
        labels = [obj['label'] for obj in qwen_result.detected_objects]
        print(f"  - Labels: {labels[:5]}...")  # Show first 5 labels

def prompt_quality_comparison():
    """Compare prompt quality between approaches."""
    print("\n=== Prompt Quality Comparison ===")
    
    image_path = "sample_image.jpg"  # Replace with actual image path
    
    if not os.path.exists(image_path):
        print(f"Please provide a valid image path. Current path: {image_path}")
        return
    
    # Test Qwen prompt generation
    qwen_config = PipelineConfig(use_qwen=True, detail_level="comprehensive")
    qwen_pipeline = AgenticSegmentationPipeline(qwen_config)
    
    # Access the engine to get generated prompts
    if hasattr(qwen_pipeline.engine, 'qwen_prompt_generator'):
        image = Image.open(image_path)
        prompts = qwen_pipeline.engine.qwen_prompt_generator.generate_enhanced_prompts(
            image, num_prompts=3, detail_level="comprehensive"
        )
        
        print("Qwen-generated prompts:")
        for i, prompt in enumerate(prompts):
            print(f"  {i+1}. {prompt.text}")
            print(f"     Confidence: {prompt.confidence:.3f}")
            print(f"     Objects: {prompt.object_categories}")
            print(f"     Reasoning: {prompt.reasoning}")
            print()
    
    # Test standard prompt generation
    standard_config = PipelineConfig(use_qwen=False)
    standard_pipeline = AgenticSegmentationPipeline(standard_config)
    
    if hasattr(standard_pipeline.engine, 'prompt_generator'):
        image = Image.open(image_path)
        standard_prompts = standard_pipeline.engine.prompt_generator.generate_prompts(image, num_prompts=3)
        
        print("Standard-generated prompts:")
        for i, prompt in enumerate(standard_prompts):
            print(f"  {i+1}. {prompt.text}")
            print(f"     Confidence: {prompt.confidence:.3f}")
            print(f"     Objects: {prompt.object_categories}")
            print()

def memory_usage_comparison():
    """Compare memory usage between approaches."""
    print("\n=== Memory Usage Comparison ===")
    
    import psutil
    import torch
    
    def get_memory_usage():
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # MB
    
    def get_gpu_memory():
        """Get GPU memory usage if available."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024  # MB
        return 0
    
    image_path = "sample_image.jpg"  # Replace with actual image path
    
    if not os.path.exists(image_path):
        print(f"Please provide a valid image path. Current path: {image_path}")
        return
    
    configs = {
        "Qwen Enhanced": PipelineConfig(use_qwen=True, use_quantization=False),
        "Qwen Quantized": PipelineConfig(use_qwen=True, use_quantization=True),
        "Standard CLIP": PipelineConfig(use_qwen=False)
    }
    
    for config_name, config in configs.items():
        print(f"\n--- Memory test: {config_name} ---")
        
        # Measure memory before initialization
        memory_before = get_memory_usage()
        gpu_memory_before = get_gpu_memory()
        
        pipeline = AgenticSegmentationPipeline(config)
        
        # Measure memory after initialization
        memory_after = get_memory_usage()
        gpu_memory_after = get_gpu_memory()
        
        print(f"RAM usage: {memory_after - memory_before:.1f} MB")
        print(f"GPU memory: {gpu_memory_after - gpu_memory_before:.1f} MB")
        
        result = pipeline.process_image(image_path)
        
        memory_peak = get_memory_usage()
        gpu_memory_peak = get_gpu_memory()
        
        print(f"Peak RAM: {memory_peak - memory_before:.1f} MB")
        print(f"Peak GPU: {gpu_memory_peak - gpu_memory_before:.1f} MB")
        
        # Clean up
        del pipeline
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def main():
    """Run all comparison examples."""
    print("Qwen vs Standard Pipeline Comparison")
    print("=" * 50)
    
    # Run comparisons
    results = compare_pipelines()
    detailed_analysis()
    prompt_quality_comparison()
    memory_usage_comparison()
    
    print("\n" + "=" * 50)
    print("Comparison completed!")
    
    # Summary recommendations
    print("\n=== Recommendations ===")
    print("1. Use Qwen Enhanced for:")
    print("   - High-quality object detection")
    print("   - Complex scenes with many objects")
    print("   - When detailed scene understanding is needed")
    print("   - Research and development applications")
    
    print("\n2. Use Standard CLIP for:")
    print("   - Fast processing requirements")
    print("   - Simple scenes with few objects")
    print("   - Resource-constrained environments")
    print("   - Production applications with speed priority")
    
    print("\n3. Use Qwen Quantized for:")
    print("   - Balance between quality and speed")
    print("   - GPU memory constraints")
    print("   - Batch processing applications")

if __name__ == "__main__":
    main()
