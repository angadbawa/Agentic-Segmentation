import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import json
import warnings

# Qwen imports
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    warnings.warn("Qwen models not available. Install with: pip install qwen-vl-utils modelscope")

# Fallback imports
if not QWEN_AVAILABLE:
    import clip
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class QwenEmbeddingResult:
    """Enhanced container for Qwen embedding results."""
    embeddings: np.ndarray
    vision_embeddings: np.ndarray
    text_embeddings: Optional[np.ndarray]
    multimodal_embeddings: np.ndarray
    features: Dict[str, np.ndarray]
    metadata: Dict[str, any]
    confidence: float

class QwenEmbeddingExtractor:
    """
    Enhanced embedding extractor using Qwen's superior multi-modal capabilities
    for better object matching and similarity analysis.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "auto",
                 use_quantization: bool = False):
        """
        Initialize the Qwen embedding extractor.
        
        Args:
            model_name: Qwen model name to use
            device: Device to run the model on
            use_quantization: Whether to use quantization for memory efficiency
        """
        self.device = self._get_device(device)
        self.model_name = model_name
        self.use_quantization = use_quantization
        
        if QWEN_AVAILABLE:
            self._load_qwen_model()
        else:
            logger.warning("Qwen not available, falling back to CLIP + SentenceTransformers")
            self._load_fallback_models()
        
        # Enhanced object database
        self.object_database = {}
        self.similarity_cache = {}
        
        logger.info(f"Initialized Qwen Embedding Extractor with {model_name} on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_qwen_model(self):
        """Load Qwen model for embedding extraction."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Load model with appropriate configuration
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None
            }
            
            if self.use_quantization and self.device == "cuda":
                model_kwargs["load_in_8bit"] = True
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name, 
                **model_kwargs
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.use_qwen = True
            
            logger.info(f"Successfully loaded Qwen model for embeddings: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            logger.info("Falling back to CLIP + SentenceTransformers")
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load fallback models (CLIP + SentenceTransformers)."""
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.use_qwen = False
        logger.info("Loaded CLIP + SentenceTransformers as fallback")
    
    def extract_multimodal_embeddings(self, 
                                     image: Image.Image,
                                     text: Optional[str] = None) -> QwenEmbeddingResult:
        """
        Extract comprehensive multimodal embeddings using Qwen.
        
        Args:
            image: PIL Image to process
            text: Optional text description
            
        Returns:
            QwenEmbeddingResult with various embeddings
        """
        if self.use_qwen:
            return self._extract_with_qwen(image, text)
        else:
            return self._extract_with_fallback(image, text)
    
    def _extract_with_qwen(self, 
                          image: Image.Image,
                          text: Optional[str] = None) -> QwenEmbeddingResult:
        """Extract embeddings using Qwen model."""
        try:
            # Prepare image for Qwen
            image_input = self._prepare_image_for_qwen(image)
            
            # Create analysis prompt
            analysis_prompt = """
            Analyze this image comprehensively. Describe all visible objects, 
            their spatial relationships, visual characteristics, and context. 
            Focus on information that would be useful for object detection and segmentation.
            """
            
            if text:
                analysis_prompt += f"\nAdditional context: {text}"
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_input},
                        {"type": "text", "text": analysis_prompt}
                    ]
                }
            ]
            
            # Process and tokenize
            text_input = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.tokenizer(
                text=text_input,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract embeddings from the model
            with torch.no_grad():
                # Get hidden states from the model
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Extract embeddings from different layers
                hidden_states = outputs.hidden_states
                
                # Use the last hidden state for embeddings
                last_hidden_state = hidden_states[-1]  # [batch_size, seq_len, hidden_size]
                
                # Pool embeddings (mean pooling)
                pooled_embeddings = torch.mean(last_hidden_state, dim=1)  # [batch_size, hidden_size]
                
                # Normalize embeddings
                normalized_embeddings = torch.nn.functional.normalize(pooled_embeddings, p=2, dim=1)
                
                # Convert to numpy
                multimodal_embeddings = normalized_embeddings.cpu().numpy().flatten()
            
            # Extract visual features
            visual_features = self._extract_visual_features(image)
            
            # Extract spatial features
            spatial_features = self._extract_spatial_features(image)
            
            # Combine all features
            combined_embeddings = np.concatenate([
                multimodal_embeddings,
                visual_features,
                spatial_features
            ])
            
            # Calculate confidence based on embedding quality
            confidence = self._calculate_embedding_confidence(multimodal_embeddings)
            
            return QwenEmbeddingResult(
                embeddings=combined_embeddings,
                vision_embeddings=multimodal_embeddings,
                text_embeddings=None,  # Integrated in multimodal
                multimodal_embeddings=multimodal_embeddings,
                features={
                    'visual': visual_features,
                    'spatial': spatial_features,
                    'multimodal': multimodal_embeddings
                },
                metadata={
                    'model': self.model_name,
                    'embedding_dim': multimodal_embeddings.shape[0],
                    'combined_dim': combined_embeddings.shape[0],
                    'has_text': text is not None,
                    'image_size': image.size
                },
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error extracting embeddings with Qwen: {e}")
            return self._extract_with_fallback(image, text)
    
    def _extract_with_fallback(self, 
                              image: Image.Image,
                              text: Optional[str] = None) -> QwenEmbeddingResult:
        """Extract embeddings using fallback models."""
        try:
            # CLIP embeddings
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                vision_embeddings = image_features.cpu().numpy().flatten()
            
            # Text embeddings
            text_embeddings = None
            if text:
                text_embeddings = self.sentence_model.encode(text)
            
            # Visual features
            visual_features = self._extract_visual_features(image)
            spatial_features = self._extract_spatial_features(image)
            
            # Combine embeddings
            embeddings_to_combine = [vision_embeddings, visual_features, spatial_features]
            if text_embeddings is not None:
                embeddings_to_combine.append(text_embeddings)
            
            combined_embeddings = np.concatenate(embeddings_to_combine)
            
            return QwenEmbeddingResult(
                embeddings=combined_embeddings,
                vision_embeddings=vision_embeddings,
                text_embeddings=text_embeddings,
                multimodal_embeddings=vision_embeddings,  # Use vision as multimodal
                features={
                    'visual': visual_features,
                    'spatial': spatial_features,
                    'clip': vision_embeddings
                },
                metadata={
                    'model': 'clip_fallback',
                    'embedding_dim': vision_embeddings.shape[0],
                    'combined_dim': combined_embeddings.shape[0],
                    'has_text': text is not None,
                    'image_size': image.size
                },
                confidence=0.7  # Lower confidence for fallback
            )
            
        except Exception as e:
            logger.error(f"Error extracting embeddings with fallback: {e}")
            return self._get_empty_embedding_result()
    
    def _prepare_image_for_qwen(self, image: Image.Image) -> str:
        """Prepare image for Qwen processing."""
        import io
        import base64
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def _extract_visual_features(self, image: Image.Image) -> np.ndarray:
        """Extract visual features from image."""
        import cv2
        
        img_array = np.array(image)
        
        features = []
        
        # Color features
        if len(img_array.shape) == 3:
            # RGB mean and std
            rgb_mean = np.mean(img_array, axis=(0, 1))
            rgb_std = np.std(img_array, axis=(0, 1))
            features.extend(rgb_mean)
            features.extend(rgb_std)
            
            # HSV features
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            hsv_mean = np.mean(hsv, axis=(0, 1))
            features.extend(hsv_mean)
        else:
            # Grayscale features
            features.extend([np.mean(img_array), np.std(img_array)])
        
        # Texture features
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        features.append(np.mean(cv2.Laplacian(gray, cv2.CV_64F)))
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features.append(np.sum(edges > 0) / (gray.shape[0] * gray.shape[1]))
        
        return np.array(features)
    
    def _extract_spatial_features(self, image: Image.Image) -> np.ndarray:
        """Extract spatial features from image."""
        width, height = image.size
        
        features = [
            width,
            height,
            width / height,  # aspect ratio
            width * height,  # area
            np.sqrt(width * height)  # diagonal
        ]
        
        return np.array(features)
    
    def _calculate_embedding_confidence(self, embeddings: np.ndarray) -> float:
        """Calculate confidence score for embeddings."""
        # Higher confidence for embeddings with good magnitude and distribution
        magnitude = np.linalg.norm(embeddings)
        variance = np.var(embeddings)
        
        # Normalize confidence (0-1)
        confidence = min(1.0, (magnitude / 10.0) + (variance / 100.0))
        return max(0.1, confidence)
    
    def compute_enhanced_similarity(self, 
                                   embeddings1: QwenEmbeddingResult,
                                   embeddings2: QwenEmbeddingResult,
                                   method: str = "multimodal") -> float:
        """
        Compute enhanced similarity between two embedding results.
        
        Args:
            embeddings1: First embedding result
            embeddings2: Second embedding result
            method: Similarity method ("multimodal", "vision", "combined")
            
        Returns:
            Similarity score
        """
        try:
            if method == "multimodal":
                emb1 = embeddings1.multimodal_embeddings
                emb2 = embeddings2.multimodal_embeddings
            elif method == "vision":
                emb1 = embeddings1.vision_embeddings
                emb2 = embeddings2.vision_embeddings
            else:  # combined
                emb1 = embeddings1.embeddings
                emb2 = embeddings2.embeddings
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            # Weight by confidence
            confidence_weight = (embeddings1.confidence + embeddings2.confidence) / 2
            weighted_similarity = similarity * confidence_weight
            
            return float(weighted_similarity)
            
        except Exception as e:
            logger.error(f"Error computing enhanced similarity: {e}")
            return 0.0
    
    def find_similar_objects_enhanced(self, 
                                    query_embedding: QwenEmbeddingResult,
                                    object_embeddings: Dict[str, QwenEmbeddingResult],
                                    top_k: int = 5,
                                    threshold: float = 0.7) -> List[Tuple[str, float, Dict[str, any]]]:
        """
        Find similar objects with enhanced matching.
        
        Args:
            query_embedding: Query embedding result
            object_embeddings: Dictionary of object_name -> embedding result
            top_k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (object_name, similarity_score, metadata) tuples
        """
        similarities = []
        
        for obj_name, obj_embedding in object_embeddings.items():
            # Compute multiple similarity scores
            multimodal_sim = self.compute_enhanced_similarity(query_embedding, obj_embedding, "multimodal")
            vision_sim = self.compute_enhanced_similarity(query_embedding, obj_embedding, "vision")
            combined_sim = self.compute_enhanced_similarity(query_embedding, obj_embedding, "combined")
            
            # Weighted combination
            final_similarity = (0.5 * multimodal_sim + 0.3 * vision_sim + 0.2 * combined_sim)
            
            if final_similarity >= threshold:
                metadata = {
                    'multimodal_similarity': multimodal_sim,
                    'vision_similarity': vision_sim,
                    'combined_similarity': combined_sim,
                    'confidence': obj_embedding.confidence
                }
                similarities.append((obj_name, final_similarity, metadata))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def create_enhanced_object_database(self, 
                                       images: List[Image.Image],
                                       object_names: List[str],
                                       descriptions: Optional[List[str]] = None) -> Dict[str, QwenEmbeddingResult]:
        """
        Create an enhanced object database using Qwen embeddings.
        
        Args:
            images: List of PIL Images
            object_names: List of object names corresponding to images
            descriptions: Optional list of text descriptions
            
        Returns:
            Dictionary mapping object names to embedding results
        """
        database = {}
        
        for i, (image, name) in enumerate(zip(images, object_names)):
            description = descriptions[i] if descriptions and i < len(descriptions) else None
            embedding_result = self.extract_multimodal_embeddings(image, description)
            database[name] = embedding_result
        
        logger.info(f"Created enhanced object database with {len(database)} objects")
        return database
    
    def _get_empty_embedding_result(self) -> QwenEmbeddingResult:
        """Get empty embedding result for error cases."""
        return QwenEmbeddingResult(
            embeddings=np.zeros(512),
            vision_embeddings=np.zeros(512),
            text_embeddings=None,
            multimodal_embeddings=np.zeros(512),
            features={},
            metadata={'error': True},
            confidence=0.0
        )
    
    def save_enhanced_database(self, 
                              database: Dict[str, QwenEmbeddingResult],
                              filepath: str):
        """Save enhanced object database to file."""
        try:
            serializable_db = {}
            for name, result in database.items():
                serializable_db[name] = {
                    'embeddings': result.embeddings.tolist(),
                    'vision_embeddings': result.vision_embeddings.tolist(),
                    'text_embeddings': result.text_embeddings.tolist() if result.text_embeddings is not None else None,
                    'multimodal_embeddings': result.multimodal_embeddings.tolist(),
                    'features': {k: v.tolist() for k, v in result.features.items()},
                    'metadata': result.metadata,
                    'confidence': result.confidence
                }
            
            with open(filepath, 'w') as f:
                json.dump(serializable_db, f)
            
            logger.info(f"Saved enhanced object database to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced database: {e}")
    
    def load_enhanced_database(self, filepath: str) -> Dict[str, QwenEmbeddingResult]:
        """Load enhanced object database from file."""
        try:
            with open(filepath, 'r') as f:
                serializable_db = json.load(f)
            
            database = {}
            for name, data in serializable_db.items():
                database[name] = QwenEmbeddingResult(
                    embeddings=np.array(data['embeddings']),
                    vision_embeddings=np.array(data['vision_embeddings']),
                    text_embeddings=np.array(data['text_embeddings']) if data['text_embeddings'] is not None else None,
                    multimodal_embeddings=np.array(data['multimodal_embeddings']),
                    features={k: np.array(v) for k, v in data['features'].items()},
                    metadata=data['metadata'],
                    confidence=data['confidence']
                )
            
            logger.info(f"Loaded enhanced object database from {filepath}")
            return database
            
        except Exception as e:
            logger.error(f"Error loading enhanced database: {e}")
            return {}
