import os
import logging
from typing import List, Tuple, Optional
from pathlib import Path
import tempfile

import gradio as gr
import numpy as np
from PIL import Image

from ..core.optimized_pipeline import OptimizedAgenticPipeline, OptimizedConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioApp:
    """Gradio application for the agentic segmentation pipeline."""
    
    def __init__(self, config_name: str = "balanced"):
        """Initialize the Gradio app."""
        self.config_name = config_name
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the pipeline."""
        try:
            config = OptimizedConfig(mode=self.config_name)
            self.pipeline = OptimizedAgenticPipeline(config)
            logger.info(f"Pipeline initialized with {self.config_name} configuration")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            self.pipeline = None
    
    def process_single_image(self, 
                           image: Image.Image,
                           mode: str,
                           confidence_threshold: float,
                           max_objects: int,
                           detail_level: str) -> Tuple[Image.Image, str, str]:
        """Process a single image."""
        if self.pipeline is None:
            return None, "Error: Pipeline not initialized", ""
        
        try:
            result = self.pipeline.process_image(
                image,
                mode=mode,
                confidence_threshold=confidence_threshold,
                max_objects=max_objects,
                detail_level=detail_level
            )
            
            visualized = self.pipeline.visualize_results(image, result, alpha=0.6)
            
            results_text = f"""
**Processing Results:**
- **Objects Detected:** {len(result.detected_objects)}
- **Processing Time:** {result.processing_time:.2f} seconds
- **Model:** {result.model_info.get('engine_type', 'unknown')}

**Detected Objects:**
"""
            
            for i, obj in enumerate(result.detected_objects, 1):
                results_text += f"""
{i}. **{obj['label']}** (confidence: {obj['confidence']:.3f})
   - Bounding Box: {obj['bbox']}
   - Area: {obj['area']} pixels
"""
            
            metadata_text = f"""
**Model Information:**
- Engine Type: {result.model_info.get('engine_type', 'unknown')}
- Qwen Model: {result.model_info.get('qwen_model', 'N/A')}
- SAM Model: {result.model_info.get('sam_model', 'unknown')}
- Device: {result.model_info.get('device', 'unknown')}

**Processing Details:**
- Mode: {result.metadata.get('mode', 'unknown')}
- Detail Level: {result.metadata.get('detail_level', 'unknown')}
- Confidence Threshold: {result.metadata.get('confidence_threshold', 'unknown')}
- Max Objects: {result.metadata.get('max_objects', 'unknown')}
"""
            
            return visualized, results_text, metadata_text
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None, f"Error: {str(e)}", ""
    
    def process_batch_images(self, 
                           images: List[Image.Image],
                           mode: str,
                           confidence_threshold: float,
                           max_objects: int,
                           detail_level: str) -> Tuple[List[Image.Image], str]:
        """Process multiple images."""
        if self.pipeline is None:
            return [], "Error: Pipeline not initialized"
        
        try:
            # Process batch
            results = self.pipeline.process_batch(
                images,
                mode=mode,
                parallel=True
            )
            
            # Generate visualizations
            visualizations = []
            for i, (image, result) in enumerate(zip(images, results)):
                visualized = self.pipeline.visualize_results(image, result, alpha=0.6)
                visualizations.append(visualized)
            
            # Create summary text
            total_objects = sum(len(result.detected_objects) for result in results)
            total_time = sum(result.processing_time for result in results)
            
            summary_text = f"""
**Batch Processing Results:**
- **Total Images:** {len(images)}
- **Total Objects Detected:** {total_objects}
- **Total Processing Time:** {total_time:.2f} seconds
- **Average Time per Image:** {total_time/len(images):.2f} seconds

**Per Image Results:**
"""
            
            for i, result in enumerate(results, 1):
                summary_text += f"""
{i}. **Image {i}:** {len(result.detected_objects)} objects detected
   - Processing Time: {result.processing_time:.2f}s
   - Objects: {', '.join([obj['label'] for obj in result.detected_objects[:3]])}
   {'...' if len(result.detected_objects) > 3 else ''}
"""
            
            return visualizations, summary_text
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return [], f"Error: {str(e)}"
    
    def add_to_database(self, 
                       image: Image.Image,
                       object_name: str,
                       description: str) -> str:
        """Add object to database."""
        if self.pipeline is None:
            return "Error: Pipeline not initialized"
        
        try:
            self.pipeline.add_object_to_database(
                image=image,
                object_name=object_name,
                description=description
            )
            
            return f"Successfully added '{object_name}' to the object database!"
            
        except Exception as e:
            logger.error(f"Error adding to database: {e}")
            return f"Error: {str(e)}"
    
    def update_config(self, config_name: str) -> str:
        """Update pipeline configuration."""
        try:
            config = get_config_preset(config_name)
            self.pipeline = AgenticSegmentationPipeline(config)
            self.config_name = config_name
            
            model_info = self.pipeline.get_model_info()
            return f"""
Configuration updated to **{config_name}**!

**Model Information:**
- Use Qwen: {model_info.get('use_qwen', False)}
- Qwen Model: {model_info.get('qwen_model', 'N/A')}
- SAM Model: {model_info.get('sam_model', 'unknown')}
- Device: {model_info.get('device', 'unknown')}
- Object Database Size: {model_info.get('object_database_size', 0)}
"""
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return f"Error: {str(e)}"
    
    def get_performance_metrics(self) -> str:
        """Get performance metrics."""
        if self.pipeline is None:
            return "Error: Pipeline not initialized"
        
        try:
            performance = self.pipeline.get_performance_summary()
            
            if not performance:
                return "No performance data available yet. Process some images to see metrics."
            
            metrics_text = "**Performance Metrics:**\n\n"
            
            for metric, stats in performance.items():
                if isinstance(stats, dict):
                    metrics_text += f"**{metric.replace('_', ' ').title()}:**\n"
                    for key, value in stats.items():
                        if isinstance(value, float):
                            metrics_text += f"- {key}: {value:.3f}\n"
                        else:
                            metrics_text += f"- {key}: {value}\n"
                    metrics_text += "\n"
                else:
                    metrics_text += f"**{metric.replace('_', ' ').title()}:** {stats}\n"
            
            return metrics_text
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return f"Error: {str(e)}"

def create_gradio_app(config_name: str = "balanced") -> gr.Blocks:
    """Create Gradio application."""
    
    app_instance = GradioApp(config_name)
    
    with gr.Blocks(
        title="Agentic Segmentation Pipeline",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🌳 Agentic Segmentation Pipeline
        
        A sophisticated AI-powered pipeline for zero-context object detection and segmentation using Qwen 2.5/3 MLLM and SAM.
        
        **Features:**
        - Zero context required - automatically generates prompts
        - Qwen 2.5/3 MLLM integration for superior understanding
        - SAM integration for precise segmentation
        - Batch processing capabilities
        - Custom object database learning
        """)
        
        with gr.Tabs():
            
            # Single Image Processing Tab
            with gr.Tab("Single Image Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        single_image = gr.Image(
                            label="Upload Image",
                            type="pil",
                            height=400
                        )
                        
                        with gr.Row():
                            single_mode = gr.Dropdown(
                                choices=["automatic", "guided", "interactive"],
                                value="automatic",
                                label="Processing Mode"
                            )
                            single_confidence = gr.Slider(
                                minimum=0.1,
                                maximum=0.9,
                                value=0.5,
                                step=0.1,
                                label="Confidence Threshold"
                            )
                        
                        with gr.Row():
                            single_max_objects = gr.Slider(
                                minimum=5,
                                maximum=50,
                                value=20,
                                step=5,
                                label="Max Objects"
                            )
                            single_detail = gr.Dropdown(
                                choices=["basic", "detailed", "comprehensive"],
                                value="comprehensive",
                                label="Detail Level"
                            )
                        
                        single_process_btn = gr.Button(
                            "Process Image",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        single_output = gr.Image(
                            label="Segmentation Results",
                            height=400
                        )
                        
                        single_results = gr.Markdown(
                            label="Results",
                            value="Upload an image and click 'Process Image' to see results."
                        )
                        
                        single_metadata = gr.Markdown(
                            label="Metadata",
                            value="Processing metadata will appear here."
                        )
                
                single_process_btn.click(
                    fn=app_instance.process_single_image,
                    inputs=[
                        single_image,
                        single_mode,
                        single_confidence,
                        single_max_objects,
                        single_detail
                    ],
                    outputs=[single_output, single_results, single_metadata]
                )
            
            # Batch Processing Tab
            with gr.Tab("📚 Batch Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_images = gr.File(
                            label="Upload Multiple Images",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        
                        with gr.Row():
                            batch_mode = gr.Dropdown(
                                choices=["automatic", "guided"],
                                value="automatic",
                                label="Processing Mode"
                            )
                            batch_confidence = gr.Slider(
                                minimum=0.1,
                                maximum=0.9,
                                value=0.5,
                                step=0.1,
                                label="Confidence Threshold"
                            )
                        
                        with gr.Row():
                            batch_max_objects = gr.Slider(
                                minimum=5,
                                maximum=50,
                                value=20,
                                step=5,
                                label="Max Objects"
                            )
                            batch_detail = gr.Dropdown(
                                choices=["basic", "detailed", "comprehensive"],
                                value="detailed",
                                label="Detail Level"
                            )
                        
                        batch_process_btn = gr.Button(
                            "Process Batch",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        batch_output = gr.Gallery(
                            label="Batch Results",
                            show_label=True,
                            elem_id="gallery",
                            columns=2,
                            rows=2,
                            height=400
                        )
                        
                        batch_results = gr.Markdown(
                            label="Batch Summary",
                            value="Upload multiple images and click 'Process Batch' to see results."
                        )
                
                batch_process_btn.click(
                    fn=app_instance.process_batch_images,
                    inputs=[
                        batch_images,
                        batch_mode,
                        batch_confidence,
                        batch_max_objects,
                        batch_detail
                    ],
                    outputs=[batch_output, batch_results]
                )
            
            # Database Management Tab
            with gr.Tab("Database Management"):
                with gr.Row():
                    with gr.Column(scale=1):
                        db_image = gr.Image(
                            label="Upload Object Image",
                            type="pil",
                            height=300
                        )
                        
                        db_name = gr.Textbox(
                            label="Object Name",
                            placeholder="e.g., 'my_custom_tree'"
                        )
                        
                        db_description = gr.Textbox(
                            label="Description (Optional)",
                            placeholder="e.g., 'A specific type of tree in my garden'",
                            lines=3
                        )
                        
                        db_add_btn = gr.Button(
                            "➕ Add to Database",
                            variant="primary"
                        )
                        
                        db_status = gr.Markdown(
                            label="Status",
                            value="Upload an image and provide a name to add objects to the database."
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        **Database Management:**
                        
                        Add custom objects to the database for better recognition:
                        
                        1. Upload an image containing the object
                        2. Provide a descriptive name
                        3. Optionally add a description
                        4. Click "Add to Database"
                        
                        The system will learn to recognize these objects in future processing.
                        """)
            
            db_add_btn.click(
                fn=app_instance.add_to_database,
                inputs=[db_image, db_name, db_description],
                outputs=[db_status]
            )
            
            # Configuration Tab
            with gr.Tab("Configuration"):
                with gr.Row():
                    with gr.Column(scale=1):
                        config_preset = gr.Dropdown(
                            choices=["fast", "balanced", "high_quality", "cpu"],
                            value="balanced",
                            label="Configuration Preset"
                        )
                        
                        config_update_btn = gr.Button(
                            "🔄 Update Configuration",
                            variant="primary"
                        )
                        
                        config_status = gr.Markdown(
                            label="Configuration Status",
                            value="Select a configuration preset and click 'Update Configuration'."
                        )
                    
                    with gr.Column(scale=1):
                        performance_btn = gr.Button(
                            "Get Performance Metrics",
                            variant="secondary"
                        )
                        
                        performance_output = gr.Markdown(
                            label="Performance Metrics",
                            value="Click 'Get Performance Metrics' to see system performance data."
                        )
                
                config_update_btn.click(
                    fn=app_instance.update_config,
                    inputs=[config_preset],
                    outputs=[config_status]
                )
                
                performance_btn.click(
                    fn=app_instance.get_performance_metrics,
                    outputs=[performance_output]
                )
        
        # Footer
        gr.Markdown("""
        ---
        **Agentic Segmentation Pipeline** - Powered by Qwen 2.5/3 MLLM and SAM
        
        For more information, visit the [GitHub repository](https://github.com/yourusername/agentic-segmentation).
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )
