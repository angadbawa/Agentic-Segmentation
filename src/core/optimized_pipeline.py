import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import time
import hashlib
from functools import lru_cache
import psutil
import warnings

# Core components
from .qwen_prompt_generator import QwenPromptGenerator
from .qwen_embedding_extractor import QwenEmbeddingExtractor
from .sam_integration import SAMSegmentation, SegmentationResult
from .feedback_system import HumanFeedbackSystem, FeedbackType, FeedbackAction
from .similarity_digitizer import SimilarityDigitizer, SimilaritySearchConfig, SimilarityMatch

logger = logging.getLogger(__name__)

@dataclass
class OptimizedConfig:
    """Simplified configuration with smart defaults."""
    mode: str = "auto"  # auto, fast, quality, custom
    device: str = "auto"
    confidence_threshold: float = 0.5
    max_objects: int = 20
    enable_caching: bool = True
    enable_feedback: bool = True
    enable_similarity: bool = True
    custom_params: Optional[Dict] = None

@dataclass
class OptimizedResult:
    """Optimized result container."""
    detected_objects: List[Dict[str, Any]]
    segmentation_masks: np.ndarray
    confidence_scores: List[float]
    processing_time: float
    mode_used: str
    cache_hit: bool = False
    feedback_session_id: Optional[str] = None
    metadata: Dict[str, Any] = None

class ModelManager:
    """Singleton pattern for shared model instances."""
    _instance = None
    _models = {}
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self, device: str = "auto"):
        """Initialize models once."""
        if self._initialized:
            return
        
        self.device = self._get_device(device)
        logger.info(f"Initializing models on device: {self.device}")
        
        self._initialized = True
    
    def get_qwen_prompt_generator(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """Get or create Qwen prompt generator."""
        key = f"qwen_prompt_{model_name}"
        if key not in self._models:
            self._models[key] = QwenPromptGenerator(
                model_name=model_name,
                device=self.device
            )
        return self._models[key]
    
    def get_qwen_embedding_extractor(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """Get or create Qwen embedding extractor."""
        key = f"qwen_embedding_{model_name}"
        if key not in self._models:
            self._models[key] = QwenEmbeddingExtractor(
                model_name=model_name,
                device=self.device
            )
        return self._models[key]
    
    def get_sam_segmentation(self, model_type: str = "vit_b"):
        """Get or create SAM segmentation."""
        key = f"sam_{model_type}"
        if key not in self._models:
            self._models[key] = SAMSegmentation(
                model_type=model_type,
                device=self.device
            )
        return self._models[key]
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

class MetricsCollector:
    """Collect and track performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'processing_times': [],
            'memory_usage': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'error_count': 0,
            'model_switches': 0
        }
    
    def track_processing(self, func):
        """Decorator to track processing metrics."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                self.metrics['processing_times'].append(time.time() - start_time)
                return result
            except Exception as e:
                self.metrics['error_count'] += 1
                logger.error(f"Error in {func.__name__}: {e}")
                raise
            finally:
                self.metrics['memory_usage'].append(
                    psutil.Process().memory_info().rss - start_memory
                )
        return wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.metrics['processing_times']:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.metrics['processing_times']),
            'max_processing_time': np.max(self.metrics['processing_times']),
            'avg_memory_usage': np.mean(self.metrics['memory_usage']),
            'cache_hit_rate': self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0,
            'error_rate': self.metrics['error_count'] / len(self.metrics['processing_times']) if self.metrics['processing_times'] else 0
        }

class OptimizedAgenticPipeline:
    """
    Optimized agentic pipeline that balances sophistication with performance.
    
    Features:
    - Shared model instances (memory efficient)
    - Adaptive complexity (simple vs complex processing)
    - Intelligent caching
    - Robust error recovery
    - Performance monitoring
    """
    
    def __init__(self, config: OptimizedConfig = None):
        """
        Initialize the optimized pipeline.
        
        Args:
            config: Configuration for the pipeline
        """
        self.config = config or OptimizedConfig()
        self.model_manager = ModelManager()
        self.metrics = MetricsCollector()
        
        # Initialize models
        self.model_manager.initialize(self.config.device)
        
        # Smart defaults based on mode
        self._setup_mode_config()
        
        # Initialize components
        self._initialize_components()
        
        # Setup caching
        if self.config.enable_caching:
            self._setup_caching()
        
        # Initialize feedback system
        if self.config.enable_feedback:
            self.feedback_system = HumanFeedbackSystem()
        else:
            self.feedback_system = None
        
        # Initialize similarity digitizer
        if self.config.enable_similarity:
            self.similarity_digitizer = SimilarityDigitizer()
        else:
            self.similarity_digitizer = None
        
        logger.info(f"Initialized Optimized Pipeline in {self.config.mode} mode")
    
    def _setup_mode_config(self):
        """Setup configuration based on mode."""
        mode_configs = {
            "auto": {
                "use_qwen": True,
                "sam_model": "vit_l",
                "confidence": 0.5,
                "max_objects": 20,
                "enable_refinement": True
            },
            "fast": {
                "use_qwen": False,
                "sam_model": "vit_b",
                "confidence": 0.6,
                "max_objects": 10,
                "enable_refinement": False
            },
            "quality": {
                "use_qwen": True,
                "sam_model": "vit_h",
                "confidence": 0.3,
                "max_objects": 50,
                "enable_refinement": True
            }
        }
        
        if self.config.mode == "custom" and self.config.custom_params:
            self.mode_config = self.config.custom_params
        else:
            self.mode_config = mode_configs.get(self.config.mode, mode_configs["auto"])
    
    def _initialize_components(self):
        """Initialize pipeline components based on mode."""
        if self.mode_config["use_qwen"]:
            self.qwen_prompt_generator = self.model_manager.get_qwen_prompt_generator()
            self.qwen_embedding_extractor = self.model_manager.get_qwen_embedding_extractor()
        else:
            self.qwen_prompt_generator = None
            self.qwen_embedding_extractor = None
        
        self.sam_segmentation = self.model_manager.get_sam_segmentation(
            self.mode_config["sam_model"]
        )
    
    def _setup_caching(self):
        """Setup intelligent caching."""
        self.image_cache = {}
        self.prompt_cache = {}
        self.max_cache_size = 100
    
    def _hash_image(self, image: Image.Image) -> str:
        """Generate hash for image caching."""
        # Convert to bytes and hash
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return hashlib.md5(buffer.getvalue()).hexdigest()
    
    def _quick_complexity_check(self, image: Image.Image) -> float:
        """Quick complexity assessment for adaptive processing."""
        width, height = image.size
        total_pixels = width * height
        
        img_array = np.array(image)
        
        # Color variance (higher = more complex)
        color_variance = np.var(img_array)
        
        # Edge density (simplified)
        gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
        edges = np.abs(np.diff(gray, axis=0)) + np.abs(np.diff(gray, axis=1))
        edge_density = np.mean(edges) / 255.0
        
        # Normalize complexity score (0-1)
        complexity = min(1.0, (color_variance / 10000) + edge_density)
        return complexity
    
    @lru_cache(maxsize=1000)
    def _cached_prompt_generation(self, image_hash: str, detail_level: str) -> str:
        """Cached prompt generation."""
        # This would be called with actual image processing
        # For now, return a placeholder
        return f"Generated prompt for {image_hash} with {detail_level}"
    
    @metrics_collector.track_processing
    def process_image(self, image: Image.Image, user_prompt: str = None, user_id: str = None) -> OptimizedResult:
        """
        Process an image with adaptive complexity and human feedback integration.
        
        Args:
            image: Input image
            user_prompt: Optional user prompt for guided processing
            user_id: Optional user identifier for feedback learning
            
        Returns:
            OptimizedResult with detection and segmentation results
        """
        start_time = time.time()
        
        # Check cache first
        cache_hit = False
        if self.config.enable_caching:
            image_hash = self._hash_image(image)
            if image_hash in self.image_cache:
                cached_result = self.image_cache[image_hash]
                cached_result.cache_hit = True
                self.metrics.metrics['cache_hits'] += 1
                return cached_result
            self.metrics.metrics['cache_misses'] += 1
        
        try:
            # Get adaptive suggestions from feedback system
            adaptive_suggestions = {}
            if self.feedback_system and user_id:
                adaptive_suggestions = self.feedback_system.get_adaptive_suggestions(image, user_id)
                logger.info(f"Adaptive suggestions: {adaptive_suggestions}")
            
            # Apply adaptive suggestions
            if adaptive_suggestions.get('recommended_mode') and adaptive_suggestions['recommended_mode'] != self.config.mode:
                logger.info(f"Adapting mode from {self.config.mode} to {adaptive_suggestions['recommended_mode']}")
                self._apply_adaptive_suggestions(adaptive_suggestions)
            
            # Adaptive complexity processing
            complexity = self._quick_complexity_check(image)
            
            if complexity < 0.3 and self.config.mode == "auto":
                # Simple processing path
                result = self._simple_process(image, user_prompt)
            else:
                # Complex processing path
                result = self._complex_process(image, user_prompt)
            
            # Apply confidence adjustments from feedback
            if adaptive_suggestions.get('confidence_adjustments'):
                result = self._apply_confidence_adjustments(result, adaptive_suggestions['confidence_adjustments'])
            
            # Cache result
            if self.config.enable_caching and len(self.image_cache) < self.max_cache_size:
                self.image_cache[image_hash] = result
            
            result.processing_time = time.time() - start_time
            result.mode_used = self.config.mode
            result.cache_hit = cache_hit
            
            # Create feedback session if enabled
            if self.feedback_system:
                feedback_session_id = self.feedback_system.collect_feedback(
                    image, 
                    asdict(result), 
                    user_id=user_id,
                    context={'complexity': complexity, 'adaptive_suggestions': adaptive_suggestions}
                )
                result.feedback_session_id = feedback_session_id
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            # Fallback to minimal processing
            return self._minimal_process(image)
    
    def _simple_process(self, image: Image.Image, user_prompt: str = None) -> OptimizedResult:
        """Simple processing path for low-complexity images."""
        logger.debug("Using simple processing path")
        
        sam_result = self.sam_segmentation.segment_automatic(
            image, 
            confidence_threshold=self.mode_config["confidence"]
        )
        
        detected_objects = []
        for i, mask in enumerate(sam_result.masks):
            if i >= self.mode_config["max_objects"]:
                break
            
            detected_objects.append({
                'label': f'object_{i}',
                'confidence': sam_result.confidence_scores[i] if i < len(sam_result.confidence_scores) else 0.5,
                'bbox': sam_result.bboxes[i] if i < len(sam_result.bboxes) else [0, 0, 100, 100],
                'mask': mask
            })
        
        return OptimizedResult(
            detected_objects=detected_objects,
            segmentation_masks=sam_result.masks,
            confidence_scores=sam_result.confidence_scores,
            processing_time=0,  # Will be set by caller
            mode_used="simple",
            metadata={'processing_path': 'simple', 'complexity': 'low'}
        )
    
    def _complex_process(self, image: Image.Image, user_prompt: str = None) -> OptimizedResult:
        """Complex processing path for high-complexity images."""
        logger.debug("Using complex processing path")
        
        if self.qwen_prompt_generator is None:
            # Fallback to simple processing if Qwen not available
            return self._simple_process(image, user_prompt)
        
        # Generate enhanced prompts using Qwen
        if user_prompt:
            # User-guided processing
            prompts = self.qwen_prompt_generator.generate_specific_prompt(
                user_prompt, image
            )
        else:
            # Automatic processing
            prompts = self.qwen_prompt_generator.generate_enhanced_prompts(
                image, num_prompts=5, detail_level="comprehensive"
            )
        
        # Process with SAM using generated prompts
        sam_result = self.sam_segmentation.segment_with_prompts(
            image, prompts, confidence_threshold=self.mode_config["confidence"]
        )
        
        # Enhanced object identification
        detected_objects = []
        for i, (mask, confidence) in enumerate(zip(sam_result.masks, sam_result.confidence_scores)):
            if i >= self.mode_config["max_objects"]:
                break
            
            # Get enhanced object information
            bbox = sam_result.bboxes[i] if i < len(sam_result.bboxes) else [0, 0, 100, 100]
            
            # Try to identify object type using embeddings
            object_label = f'object_{i}'
            object_embedding = None
            if self.qwen_embedding_extractor:
                try:
                    object_embedding = self.qwen_embedding_extractor.extract_object_embedding(
                        image, bbox
                    )
                    # Simple object classification (would be more sophisticated in practice)
                    object_label = self._classify_object(object_embedding)
                except Exception as e:
                    logger.warning(f"Object classification failed: {e}")
            
            # Add to similarity digitizer if enabled
            feature_id = None
            if self.similarity_digitizer and object_embedding is not None:
                try:
                    feature_id = self.similarity_digitizer.add_feature(
                        image=image,
                        bbox=bbox,
                        label=object_label,
                        confidence=confidence,
                        embedding=object_embedding,
                        metadata={'processing_mode': 'complex', 'timestamp': time.time()}
                    )
                except Exception as e:
                    logger.warning(f"Failed to add feature to similarity digitizer: {e}")
            
            detected_objects.append({
                'label': object_label,
                'confidence': confidence,
                'bbox': bbox,
                'mask': mask,
                'embedding': object_embedding,
                'feature_id': feature_id
            })
        
        # Refinement if enabled
        if self.mode_config["enable_refinement"]:
            detected_objects = self._refine_detections(image, detected_objects)
        
        return OptimizedResult(
            detected_objects=detected_objects,
            segmentation_masks=sam_result.masks,
            confidence_scores=sam_result.confidence_scores,
            processing_time=0,  # Will be set by caller
            mode_used="complex",
            metadata={
                'processing_path': 'complex',
                'prompts_generated': len(prompts),
                'refinement_applied': self.mode_config["enable_refinement"]
            }
        )
    
    def _minimal_process(self, image: Image.Image) -> OptimizedResult:
        """Minimal processing fallback."""
        logger.warning("Using minimal processing fallback")
        
        sam_result = self.sam_segmentation.segment_automatic(
            image, confidence_threshold=0.7
        )
        
        detected_objects = []
        for i, mask in enumerate(sam_result.masks[:5]):  # Limit to 5 objects
            detected_objects.append({
                'label': f'object_{i}',
                'confidence': 0.5,
                'bbox': [0, 0, 100, 100],
                'mask': mask
            })
        
        return OptimizedResult(
            detected_objects=detected_objects,
            segmentation_masks=sam_result.masks,
            confidence_scores=[0.5] * len(detected_objects),
            processing_time=0,
            mode_used="minimal",
            metadata={'processing_path': 'minimal', 'fallback': True}
        )
    
    def _classify_object(self, embedding: np.ndarray) -> str:
        """Simple object classification based on embedding."""
        # This would be more sophisticated in practice
        # For now, return a generic label
        return "detected_object"
    
    def _refine_detections(self, image: Image.Image, detections: List[Dict]) -> List[Dict]:
        """Refine detection results."""
        refined = []
        for detection in detections:
            if detection['confidence'] >= self.mode_config["confidence"]:
                refined.append(detection)
        
        return refined
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.metrics.get_stats()
    
    def clear_cache(self):
        """Clear all caches."""
        if hasattr(self, 'image_cache'):
            self.image_cache.clear()
        if hasattr(self, 'prompt_cache'):
            self.prompt_cache.clear()
        logger.info("Cache cleared")
    
    def switch_mode(self, new_mode: str):
        """Switch processing mode dynamically."""
        if new_mode not in ["auto", "fast", "quality", "custom"]:
            raise ValueError(f"Invalid mode: {new_mode}")
        
        self.config.mode = new_mode
        self._setup_mode_config()
        self._initialize_components()
        
        logger.info(f"Switched to {new_mode} mode")
    
    def _apply_adaptive_suggestions(self, suggestions: Dict[str, Any]):
        """Apply adaptive suggestions from feedback system."""
        if suggestions.get('recommended_mode'):
            self.switch_mode(suggestions['recommended_mode'])
        
        # Apply other adaptive configurations
        if suggestions.get('expected_objects'):
            # Could be used to prioritize certain object types
            logger.info(f"Expected objects: {suggestions['expected_objects']}")
    
    def _apply_confidence_adjustments(self, result: OptimizedResult, adjustments: Dict[str, float]) -> OptimizedResult:
        """Apply confidence adjustments based on user feedback."""
        for i, obj in enumerate(result.detected_objects):
            object_type = obj.get('label', 'unknown')
            if object_type in adjustments:
                adjustment = adjustments[object_type]
                # Apply adjustment to confidence
                new_confidence = max(0.0, min(1.0, obj['confidence'] + adjustment))
                result.detected_objects[i]['confidence'] = new_confidence
                result.confidence_scores[i] = new_confidence
                logger.debug(f"Adjusted confidence for {object_type}: {obj['confidence']} -> {new_confidence}")
        
        return result
    
    def add_feedback(self, 
                    feedback_session_id: str,
                    object_feedbacks: List[Dict[str, Any]] = None,
                    processing_feedback: Dict[str, Any] = None):
        """
        Add human feedback to improve future processing.
        
        Args:
            feedback_session_id: Session ID from process_image result
            object_feedbacks: List of object-specific feedback
            processing_feedback: Overall processing feedback
        """
        if not self.feedback_system:
            logger.warning("Feedback system not enabled")
            return
        
        try:
            # Add object feedbacks
            if object_feedbacks:
                for obj_feedback in object_feedbacks:
                    self.feedback_system.add_object_feedback(
                        session_id=feedback_session_id,
                        object_id=obj_feedback['object_id'],
                        action=FeedbackAction(obj_feedback['action']),
                        original_detection=obj_feedback['original_detection'],
                        corrected_detection=obj_feedback.get('corrected_detection'),
                        user_comment=obj_feedback.get('user_comment'),
                        confidence_rating=obj_feedback.get('confidence_rating')
                    )
            
            # Add processing feedback
            if processing_feedback:
                self.feedback_system.add_processing_feedback(
                    session_id=feedback_session_id,
                    feedback_type=FeedbackType(processing_feedback['feedback_type']),
                    overall_rating=processing_feedback.get('overall_rating'),
                    processing_time_acceptable=processing_feedback.get('processing_time_acceptable'),
                    preferred_mode=processing_feedback.get('preferred_mode'),
                    user_comment=processing_feedback.get('user_comment'),
                    suggestions=processing_feedback.get('suggestions')
                )
            
            # Learn from feedback
            learning_insights = self.feedback_system.learn_from_feedback(feedback_session_id)
            logger.info(f"Learned from feedback: {learning_insights}")
            
        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
    
    def get_feedback_statistics(self, user_id: str = None) -> Dict[str, Any]:
        """Get feedback statistics for analysis."""
        if not self.feedback_system:
            return {}
        
        return self.feedback_system.get_feedback_statistics(user_id)
    
    def get_adaptive_suggestions(self, image: Image.Image, user_id: str = None) -> Dict[str, Any]:
        """Get adaptive suggestions based on user feedback history."""
        if not self.feedback_system:
            return {}
        
        return self.feedback_system.get_adaptive_suggestions(image, user_id)
    
    def find_similar_features(self, 
                            feature_id: str,
                            similarity_threshold: float = 0.7,
                            max_results: int = 50) -> SimilarityMatch:
        """
        Find features similar to the specified feature.
        
        Args:
            feature_id: ID of the target feature
            similarity_threshold: Minimum similarity threshold
            max_results: Maximum number of results
            
        Returns:
            SimilarityMatch with similar features and scores
        """
        if not self.similarity_digitizer:
            raise ValueError("Similarity digitizer not enabled")
        
        config = SimilaritySearchConfig(
            similarity_threshold=similarity_threshold,
            max_results=max_results
        )
        
        return self.similarity_digitizer.find_similar_features(feature_id, config)
    
    def search_by_class(self, 
                       class_label: str,
                       max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Find all features belonging to a specific class.
        
        Args:
            class_label: Class label to search for
            max_results: Maximum number of results
            
        Returns:
            List of features belonging to the class
        """
        if not self.similarity_digitizer:
            raise ValueError("Similarity digitizer not enabled")
        
        config = SimilaritySearchConfig(max_results=max_results)
        features = self.similarity_digitizer.search_by_class(class_label, config)
        
        # Convert to dictionary format
        return [
            {
                'feature_id': f.feature_id,
                'image_path': f.image_path,
                'bbox': f.bbox,
                'label': f.label,
                'confidence': f.confidence,
                'metadata': f.metadata,
                'timestamp': f.timestamp.isoformat()
            }
            for f in features
        ]
    
    def advanced_similarity_search(self, 
                                 query_embedding: np.ndarray,
                                 class_filter: Optional[str] = None,
                                 similarity_threshold: float = 0.7,
                                 max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Advanced similarity search using a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            class_filter: Optional class filter
            similarity_threshold: Minimum similarity threshold
            max_results: Maximum number of results
            
        Returns:
            List of (feature, similarity_score) tuples
        """
        if not self.similarity_digitizer:
            raise ValueError("Similarity digitizer not enabled")
        
        config = SimilaritySearchConfig(
            similarity_threshold=similarity_threshold,
            max_results=max_results
        )
        
        results = self.similarity_digitizer.advanced_similarity_search(
            query_embedding, class_filter, config
        )
        
        # Convert to dictionary format
        return [
            {
                'feature': {
                    'feature_id': f.feature_id,
                    'image_path': f.image_path,
                    'bbox': f.bbox,
                    'label': f.label,
                    'confidence': f.confidence,
                    'metadata': f.metadata,
                    'timestamp': f.timestamp.isoformat()
                },
                'similarity_score': score
            }
            for f, score in results
        ]
    
    def visualize_similarity_results(self, 
                                   feature_id: str,
                                   save_path: Optional[str] = None) -> str:
        """
        Create visualization of similarity results for a feature.
        
        Args:
            feature_id: ID of the target feature
            save_path: Optional path to save visualization
            
        Returns:
            Path to saved visualization
        """
        if not self.similarity_digitizer:
            raise ValueError("Similarity digitizer not enabled")
        
        # Find similar features
        similarity_match = self.find_similar_features(feature_id)
        
        # Create visualization
        return self.similarity_digitizer.visualize_similarity_results(
            similarity_match, save_path
        )
    
    def get_similarity_statistics(self) -> Dict[str, Any]:
        """Get similarity digitizer statistics."""
        if not self.similarity_digitizer:
            return {}
        
        return self.similarity_digitizer.get_database_statistics()
    
    def export_similarity_features(self, 
                                 output_path: str,
                                 class_filter: Optional[str] = None,
                                 format: str = "json") -> str:
        """
        Export similarity features to file.
        
        Args:
            output_path: Output file path
            class_filter: Optional class filter
            format: Export format ("json", "csv")
            
        Returns:
            Path to exported file
        """
        if not self.similarity_digitizer:
            raise ValueError("Similarity digitizer not enabled")
        
        return self.similarity_digitizer.export_features(output_path, class_filter, format)

# Global model manager instance
model_manager = ModelManager()

# Decorator for metrics collection
def metrics_collector(func):
    """Decorator to collect metrics."""
    def wrapper(self, *args, **kwargs):
        return self.metrics.track_processing(func)(self, *args, **kwargs)
    return wrapper
