#!/usr/bin/env python3
"""
Command Line Interface for Agentic Segmentation Pipeline
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from core.optimized_pipeline import OptimizedAgenticPipeline, OptimizedConfig

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Agentic Segmentation Pipeline - Zero-context object detection and segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli process image.jpg

  python -m src.cli process image.jpg --config high_quality --mode guided

  python -m src.cli batch images/ --output results/

  # Start web interface
  python -m src.cli web --interface flask

  # Start Gradio interface
  python -m src.cli web --interface gradio
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process a single image')
    process_parser.add_argument('image', help='Path to input image')
    process_parser.add_argument('--output', '-o', help='Output directory for results')
    process_parser.add_argument('--config', '-c', choices=['fast', 'balanced', 'high_quality', 'cpu'], 
                               default='balanced', help='Configuration preset')
    process_parser.add_argument('--mode', '-m', choices=['automatic', 'guided', 'interactive'], 
                               default='automatic', help='Processing mode')
    process_parser.add_argument('--confidence', type=float, default=0.5, 
                               help='Confidence threshold (0.1-0.9)')
    process_parser.add_argument('--max-objects', type=int, default=20, 
                               help='Maximum number of objects to detect')
    process_parser.add_argument('--detail-level', choices=['basic', 'detailed', 'comprehensive'], 
                               default='comprehensive', help='Detail level for analysis')
    process_parser.add_argument('--save-viz', action='store_true', 
                               help='Save visualization image')
    process_parser.add_argument('--verbose', '-v', action='store_true', 
                               help='Verbose output')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process multiple images')
    batch_parser.add_argument('input_dir', help='Input directory containing images')
    batch_parser.add_argument('--output', '-o', help='Output directory for results')
    batch_parser.add_argument('--config', '-c', choices=['fast', 'balanced', 'high_quality', 'cpu'], 
                             default='balanced', help='Configuration preset')
    batch_parser.add_argument('--mode', '-m', choices=['automatic', 'guided'], 
                             default='automatic', help='Processing mode')
    batch_parser.add_argument('--parallel', action='store_true', default=True, 
                             help='Process images in parallel')
    batch_parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp'], 
                             help='Image file extensions to process')
    batch_parser.add_argument('--verbose', '-v', action='store_true', 
                             help='Verbose output')
    
    # Web command
    web_parser = subparsers.add_parser('web', help='Start web interface')
    web_parser.add_argument('--interface', '-i', choices=['flask', 'gradio'], 
                           default='flask', help='Web interface to use')
    web_parser.add_argument('--config', '-c', choices=['fast', 'balanced', 'high_quality', 'cpu'], 
                           default='balanced', help='Configuration preset')
    web_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    web_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    web_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Database command
    db_parser = subparsers.add_parser('database', help='Database management')
    db_subparsers = db_parser.add_subparsers(dest='db_command', help='Database commands')
    
    # Add to database
    db_add_parser = db_subparsers.add_parser('add', help='Add object to database')
    db_add_parser.add_argument('image', help='Path to image containing the object')
    db_add_parser.add_argument('name', help='Name of the object')
    db_add_parser.add_argument('--description', help='Description of the object')
    db_add_parser.add_argument('--bbox', help='Bounding box as "x1,y1,x2,y2"')
    
    # Save database
    db_save_parser = db_subparsers.add_parser('save', help='Save database to file')
    db_save_parser.add_argument('--output', '-o', default='object_database.json', 
                               help='Output file path')
    
    # Load database
    db_load_parser = db_subparsers.add_parser('load', help='Load database from file')
    db_load_parser.add_argument('input_file', help='Input database file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'process':
            process_single_image(args)
        elif args.command == 'batch':
            process_batch_images(args)
        elif args.command == 'web':
            start_web_interface(args)
        elif args.command == 'database':
            handle_database_command(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose if hasattr(args, 'verbose') else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def process_single_image(args):
    """Process a single image."""
    print(f"Loading pipeline with {args.config} configuration...")
    config = OptimizedConfig(mode=args.config)
    pipeline = OptimizedAgenticPipeline(config)
    
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        sys.exit(1)
    
    print(f"Processing image: {args.image}")
    
    result = pipeline.process_image(
        args.image,
        mode=args.mode,
        confidence_threshold=args.confidence,
        max_objects=args.max_objects,
        detail_level=args.detail_level
    )
    
    print(f"\nResults:")
    print(f"  Objects detected: {len(result.detected_objects)}")
    print(f"  Processing time: {result.processing_time:.2f} seconds")
    print(f"  Model: {result.model_info.get('engine_type', 'unknown')}")
    
    if result.detected_objects:
        print(f"\nDetected objects:")
        for i, obj in enumerate(result.detected_objects, 1):
            print(f"  {i}. {obj['label']} (confidence: {obj['confidence']:.3f})")
    
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.save_viz:
            from PIL import Image
            image = Image.open(args.image)
            visualized = pipeline.visualize_results(image, result, alpha=0.6)
            viz_path = output_dir / f"visualization_{Path(args.image).stem}.png"
            visualized.save(viz_path)
            print(f"  Visualization saved: {viz_path}")
        
        import json
        results_data = {
            'image': args.image,
            'objects_detected': len(result.detected_objects),
            'processing_time': result.processing_time,
            'objects': [
                {
                    'label': obj['label'],
                    'confidence': obj['confidence'],
                    'bbox': obj['bbox'],
                    'area': obj['area']
                }
                for obj in result.detected_objects
            ],
            'model_info': result.model_info,
            'metadata': result.metadata
        }
        
        results_path = output_dir / f"results_{Path(args.image).stem}.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"  Results saved: {results_path}")

def process_batch_images(args):
    """Process multiple images."""
    print(f"Loading pipeline with {args.config} configuration...")
    config = OptimizedConfig(mode=args.config)
    pipeline = OptimizedAgenticPipeline(config)
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' not found.")
        sys.exit(1)
    
    # Find image files
    image_files = []
    for ext in args.extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"Error: No image files found in '{input_dir}' with extensions {args.extensions}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to process")
    
    # Process batch
    results = pipeline.process_batch(
        [str(f) for f in image_files],
        mode=args.mode,
        parallel=args.parallel
    )
    
    # Print summary
    total_objects = sum(len(result.detected_objects) for result in results)
    total_time = sum(result.processing_time for result in results)
    
    print(f"\nBatch Processing Results:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Total objects detected: {total_objects}")
    print(f"  Total processing time: {total_time:.2f} seconds")
    print(f"  Average time per image: {total_time/len(image_files):.2f} seconds")
    
    # Save results if output directory specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        batch_results = {
            'total_images': len(image_files),
            'total_objects': total_objects,
            'total_time': total_time,
            'results': []
        }
        
        for i, (image_file, result) in enumerate(zip(image_files, results)):
            batch_results['results'].append({
                'image': str(image_file),
                'objects_detected': len(result.detected_objects),
                'processing_time': result.processing_time,
                'objects': [
                    {
                        'label': obj['label'],
                        'confidence': obj['confidence']
                    }
                    for obj in result.detected_objects
                ]
            })
        
        results_path = output_dir / "batch_results.json"
        with open(results_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        print(f"  Batch results saved: {results_path}")

def start_web_interface(args):
    """Start web interface."""
    if args.interface == 'flask':
        from web.flask_app import create_app
        app = create_app(args.config)
        print(f"Starting Flask app on http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)
    elif args.interface == 'gradio':
        from web.gradio_app import create_gradio_app
        demo = create_gradio_app(args.config)
        print(f"Starting Gradio app on http://{args.host}:{args.port}")
        demo.launch(server_name=args.host, server_port=args.port, debug=args.debug)

def handle_database_command(args):
    """Handle database commands."""
    config = OptimizedConfig(mode='balanced')
    pipeline = OptimizedAgenticPipeline(config)
    
    if args.db_command == 'add':
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found.")
            sys.exit(1)
        
        bbox = None
        if args.bbox:
            try:
                bbox = [int(x) for x in args.bbox.split(',')]
                if len(bbox) != 4:
                    raise ValueError("Bounding box must have 4 values")
            except ValueError as e:
                print(f"Error: Invalid bounding box format. {e}")
                sys.exit(1)
        
        pipeline.add_object_to_database(
            image=args.image,
            object_name=args.name,
            description=args.description,
            bbox=bbox
        )
        print(f"Successfully added '{args.name}' to the object database!")
    
    elif args.db_command == 'save':
        pipeline.save_database(args.output)
        print(f"Database saved to: {args.output}")
    
    elif args.db_command == 'load':
        if not os.path.exists(args.input_file):
            print(f"Error: Database file '{args.input_file}' not found.")
            sys.exit(1)
        
        pipeline.load_database(args.input_file)
        print(f"Database loaded from: {args.input_file}")
    
    else:
        print("Error: Unknown database command.")
        sys.exit(1)

if __name__ == '__main__':
    main()
