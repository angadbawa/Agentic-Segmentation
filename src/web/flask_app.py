import os
import io
import base64
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_file, Response
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

from ..core.optimized_pipeline import OptimizedAgenticPipeline, OptimizedConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app(config_name: str = "balanced") -> Flask:
    """Create and configure Flask application."""
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Create upload and output directories
    upload_dir = Path("uploads")
    output_dir = Path("outputs")
    upload_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    app.config['UPLOAD_FOLDER'] = str(upload_dir)
    app.config['OUTPUT_FOLDER'] = str(output_dir)
    
    # Initialize pipeline
    try:
        config = OptimizedConfig(mode=config_name)
        pipeline = OptimizedAgenticPipeline(config)
        logger.info(f"Pipeline initialized with {config_name} configuration")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        pipeline = None
    
    @app.route('/')
    def index():
        """Main page."""
        return render_template('index.html')
    
    @app.route('/health')
    def health():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'pipeline_ready': pipeline is not None
        })
    
    @app.route('/api/config', methods=['GET'])
    def get_config():
        """Get current configuration."""
        if pipeline is None:
            return jsonify({'error': 'Pipeline not initialized'}), 500
        
        return jsonify({
            'model_info': pipeline.get_model_info(),
            'config_presets': ['fast', 'balanced', 'high_quality', 'cpu']
        })
    
    @app.route('/api/config', methods=['POST'])
    def update_config():
        """Update configuration."""
        try:
            config_data = request.json
            config_name = config_data.get('preset', 'balanced')
            
            global pipeline
            config = get_config_preset(config_name)
            pipeline = AgenticSegmentationPipeline(config)
            
            return jsonify({
                'message': f'Configuration updated to {config_name}',
                'model_info': pipeline.get_model_info()
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/process', methods=['POST'])
    def process_image():
        """Process uploaded image."""
        if pipeline is None:
            return jsonify({'error': 'Pipeline not initialized'}), 500
        
        try:
            mode = request.form.get('mode', 'automatic')
            confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
            max_objects = int(request.form.get('max_objects', 20))
            detail_level = request.form.get('detail_level', 'comprehensive')
            
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No image file selected'}), 400
            
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"Processing image: {filename}")
            result = pipeline.process_image(
                filepath,
                mode=mode,
                confidence_threshold=confidence_threshold,
                max_objects=max_objects,
                detail_level=detail_level
            )
            
            image = Image.open(filepath)
            visualized = pipeline.visualize_results(image, result, alpha=0.6)
            
            viz_filename = f"viz_{filename}"
            viz_path = os.path.join(app.config['OUTPUT_FOLDER'], viz_filename)
            visualized.save(viz_path)
            
            # Prepare response
            response_data = {
                'success': True,
                'filename': filename,
                'visualization': viz_filename,
                'processing_time': result.processing_time,
                'objects_detected': len(result.detected_objects),
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
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/process_batch', methods=['POST'])
    def process_batch():
        """Process multiple images."""
        if pipeline is None:
            return jsonify({'error': 'Pipeline not initialized'}), 500
        
        try:
            # Get parameters
            mode = request.form.get('mode', 'automatic')
            parallel = request.form.get('parallel', 'true').lower() == 'true'
            
            # Handle multiple files
            files = request.files.getlist('images')
            if not files:
                return jsonify({'error': 'No image files provided'}), 400
            
            # Save uploaded files
            filepaths = []
            for file in files:
                if file.filename:
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{timestamp}_{filename}"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    filepaths.append(filepath)
            
            # Process batch
            logger.info(f"Processing batch of {len(filepaths)} images")
            results = pipeline.process_batch(filepaths, mode=mode, parallel=parallel)
            
            # Prepare response
            batch_results = []
            for i, result in enumerate(results):
                batch_results.append({
                    'filename': os.path.basename(filepaths[i]),
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
            
            return jsonify({
                'success': True,
                'total_images': len(filepaths),
                'results': batch_results
            })
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/visualize/<filename>')
    def get_visualization(filename):
        """Get visualization image."""
        try:
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            if os.path.exists(filepath):
                return send_file(filepath, mimetype='image/png')
            else:
                return jsonify({'error': 'Visualization not found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/upload', methods=['POST'])
    def upload_image():
        """Upload image for processing."""
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No image file selected'}), 400
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'message': 'Image uploaded successfully'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/database/add', methods=['POST'])
    def add_to_database():
        """Add object to database."""
        if pipeline is None:
            return jsonify({'error': 'Pipeline not initialized'}), 500
        
        try:
            object_name = request.form.get('object_name')
            description = request.form.get('description', '')
            bbox = request.form.get('bbox')  # JSON string
            
            if not object_name:
                return jsonify({'error': 'Object name is required'}), 400
            
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No image file selected'}), 400
            
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Parse bounding box if provided
            bbox_list = None
            if bbox:
                try:
                    bbox_list = json.loads(bbox)
                except json.JSONDecodeError:
                    return jsonify({'error': 'Invalid bounding box format'}), 400
            
            pipeline.add_object_to_database(
                image=filepath,
                object_name=object_name,
                description=description,
                bbox=bbox_list
            )
            
            return jsonify({
                'success': True,
                'message': f'Object "{object_name}" added to database'
            })
            
        except Exception as e:
            logger.error(f"Error adding to database: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/database/save', methods=['POST'])
    def save_database():
        """Save object database."""
        if pipeline is None:
            return jsonify({'error': 'Pipeline not initialized'}), 500
        
        try:
            filename = request.json.get('filename', 'object_database.json')
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            
            pipeline.save_database(filepath)
            
            return jsonify({
                'success': True,
                'message': f'Database saved to {filename}'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/performance', methods=['GET'])
    def get_performance():
        """Get performance metrics."""
        if pipeline is None:
            return jsonify({'error': 'Pipeline not initialized'}), 500
        
        try:
            performance = pipeline.get_performance_summary()
            return jsonify(performance)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8000, debug=False)
