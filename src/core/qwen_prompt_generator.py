import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import json
import warnings

# Qwen imports
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    warnings.warn("Qwen models not available. Install with: pip install qwen-vl-utils modelscope")

if not QWEN_AVAILABLE:
    import clip

logger = logging.getLogger(__name__)

@dataclass
class QwenPromptResult:
    """Enhanced container for Qwen-generated prompts."""
    text: str
    confidence: float
    object_categories: List[str]
    spatial_context: str
    visual_features: List[str]
    detailed_description: str
    scene_understanding: Dict[str, any]
    reasoning: str

class QwenPromptGenerator:
    """
    Enhanced prompt generator using Qwen 2.5/3 MLLM for superior
    vision-language understanding and prompt generation.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "auto",
                 use_quantization: bool = False):
        """
        Initialize the Qwen prompt generator.
        
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
            logger.warning("Qwen not available, falling back to CLIP")
            self._load_clip_fallback()
        
        # Enhanced object categories for better understanding
        self.object_categories = [
            # People and animals
            "person", "man", "woman", "child", "baby", "dog", "cat", "bird", "horse", 
            "cow", "sheep", "pig", "chicken", "fish", "elephant", "lion", "tiger",
            
            # Vehicles
            "car", "truck", "bus", "motorcycle", "bicycle", "train", "airplane", 
            "boat", "ship", "helicopter", "taxi", "van", "suv",
            
            # Buildings and structures
            "building", "house", "apartment", "office", "school", "hospital", 
            "church", "bridge", "tower", "wall", "fence", "gate",
            
            # Nature and environment
            "tree", "flower", "grass", "mountain", "hill", "river", "lake", 
            "ocean", "beach", "forest", "field", "sky", "cloud", "sun", "moon",
            
            # Objects and items
            "chair", "table", "bed", "sofa", "lamp", "book", "phone", "laptop", 
            "computer", "television", "camera", "bag", "backpack", "hat", "shoes",
            
            # Food and drinks
            "apple", "banana", "orange", "bread", "cake", "pizza", "burger", 
            "coffee", "water", "bottle", "cup", "plate", "bowl",
            
            # Signs and text
            "sign", "traffic light", "stop sign", "billboard", "text", "logo"
        ]
        
        logger.info(f"Initialized Qwen Prompt Generator with {model_name} on {self.device}")
    
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
        """Load Qwen model and tokenizer."""
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
            
            logger.info(f"Successfully loaded Qwen model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            logger.info("Falling back to CLIP")
            self._load_clip_fallback()
    
    def _load_clip_fallback(self):
        """Load CLIP as fallback."""
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.use_qwen = False
        logger.info("Loaded CLIP as fallback")
    
    def generate_enhanced_prompts(self, 
                                 image: Image.Image, 
                                 num_prompts: int = 5,
                                 detail_level: str = "comprehensive") -> List[QwenPromptResult]:
        """
        Generate enhanced prompts using Qwen's superior understanding.
        
        Args:
            image: PIL Image to analyze
            num_prompts: Number of prompts to generate
            detail_level: Level of detail ("basic", "detailed", "comprehensive")
            
        Returns:
            List of QwenPromptResult objects
        """
        if self.use_qwen:
            return self._generate_with_qwen(image, num_prompts, detail_level)
        else:
            return self._generate_with_clip_fallback(image, num_prompts)
    
    def _generate_with_qwen(self, 
                           image: Image.Image,
                           num_prompts: int,
                           detail_level: str) -> List[QwenPromptResult]:
        """Generate prompts using Qwen model."""
        try:
            # Prepare the image for Qwen
            image_input = self._prepare_image_for_qwen(image)
            
            # Create comprehensive analysis prompts
            analysis_prompts = self._create_analysis_prompts(detail_level)
            
            results = []
            
            for i, prompt_template in enumerate(analysis_prompts[:num_prompts]):
                # Generate response using Qwen
                response = self._query_qwen(image_input, prompt_template)
                
                # Parse the response
                parsed_result = self._parse_qwen_response(response, prompt_template)
                
                # Create enhanced prompt for segmentation
                segmentation_prompt = self._create_segmentation_prompt(parsed_result)
                
                result = QwenPromptResult(
                    text=segmentation_prompt,
                    confidence=parsed_result.get('confidence', 0.8),
                    object_categories=parsed_result.get('objects', []),
                    spatial_context=parsed_result.get('spatial_context', ''),
                    visual_features=parsed_result.get('visual_features', []),
                    detailed_description=parsed_result.get('description', ''),
                    scene_understanding=parsed_result.get('scene_analysis', {}),
                    reasoning=parsed_result.get('reasoning', '')
                )
                
                results.append(result)
            
            logger.info(f"Generated {len(results)} enhanced prompts with Qwen")
            return results
            
        except Exception as e:
            logger.error(f"Error generating prompts with Qwen: {e}")
            return self._generate_with_clip_fallback(image, num_prompts)
    
    def _prepare_image_for_qwen(self, image: Image.Image) -> str:
        """Prepare image for Qwen processing."""
        import io
        import base64
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def _create_analysis_prompts(self, detail_level: str) -> List[str]:
        """Create analysis prompts for different detail levels."""
        if detail_level == "basic":
            return [
                "Analyze this image and identify the main objects. List the most prominent objects you can see.",
                "What are the primary visual elements in this image? Focus on objects that would be important for segmentation.",
                "Describe the main subjects and objects in this scene."
            ]
        elif detail_level == "detailed":
            return [
                "Provide a comprehensive analysis of this image. Identify all visible objects, their spatial relationships, and visual characteristics. Focus on elements that would be important for object detection and segmentation.",
                "Analyze this image in detail. What objects can you see? Where are they located? What are their visual properties? Consider lighting, colors, textures, and spatial arrangement.",
                "Examine this image thoroughly. List all objects, describe their appearance, position, and context. Pay attention to details that would help with accurate segmentation."
            ]
        else:  # comprehensive
            return [
                "Conduct a thorough visual analysis of this image. Identify all objects, their precise locations, visual characteristics, spatial relationships, and contextual information. Provide detailed descriptions that would enable accurate object detection and segmentation. Consider lighting conditions, color schemes, textures, and any occlusions or partial visibility.",
                "Perform an exhaustive examination of this image. Catalog every visible object, describe their appearance in detail, analyze their spatial arrangement, and provide contextual understanding. Focus on information that would be crucial for precise segmentation tasks, including object boundaries, overlapping elements, and visual complexity.",
                "Analyze this image with maximum detail and precision. Identify all objects, their exact positions, visual properties, relationships to other elements, and scene context. Provide comprehensive descriptions that would support advanced segmentation algorithms, considering factors like lighting, shadows, reflections, and material properties."
            ]
    
    def _query_qwen(self, image_input: str, prompt: str) -> str:
        """Query Qwen model with image and prompt."""
        try:
            # Prepare messages for Qwen
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_input},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.tokenizer(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # Decode response
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying Qwen: {e}")
            return "Error in analysis"
    
    def _parse_qwen_response(self, response: str, original_prompt: str) -> Dict[str, any]:
        """Parse Qwen response into structured data."""
        try:
            # Extract objects mentioned in the response
            objects = []
            for category in self.object_categories:
                if category.lower() in response.lower():
                    objects.append(category)
            
            # Extract spatial context
            spatial_keywords = ["left", "right", "center", "top", "bottom", "foreground", "background", "middle"]
            spatial_context = []
            for keyword in spatial_keywords:
                if keyword in response.lower():
                    spatial_context.append(keyword)
            
            # Extract visual features
            visual_keywords = ["bright", "dark", "colorful", "large", "small", "sharp", "blurry", "textured"]
            visual_features = []
            for keyword in visual_keywords:
                if keyword in response.lower():
                    visual_features.append(keyword)
            
            # Calculate confidence based on response quality
            confidence = min(0.9, 0.5 + len(objects) * 0.1 + len(visual_features) * 0.05)
            
            return {
                'objects': objects,
                'spatial_context': ' '.join(spatial_context) if spatial_context else 'center',
                'visual_features': visual_features,
                'description': response,
                'confidence': confidence,
                'scene_analysis': {
                    'object_count': len(objects),
                    'complexity': 'high' if len(objects) > 5 else 'medium' if len(objects) > 2 else 'low',
                    'spatial_awareness': len(spatial_context) > 0
                },
                'reasoning': f"Analysis based on {original_prompt[:50]}..."
            }
            
        except Exception as e:
            logger.error(f"Error parsing Qwen response: {e}")
            return {
                'objects': [],
                'spatial_context': 'unknown',
                'visual_features': [],
                'description': response,
                'confidence': 0.3,
                'scene_analysis': {},
                'reasoning': 'Error in parsing'
            }
    
    def _create_segmentation_prompt(self, parsed_result: Dict[str, any]) -> str:
        """Create segmentation-specific prompt from parsed results."""
        objects = parsed_result.get('objects', [])
        spatial_context = parsed_result.get('spatial_context', '')
        visual_features = parsed_result.get('visual_features', [])
        
        if not objects:
            return "Detect and segment all objects in this image"
        
        if len(objects) == 1:
            prompt = f"Find and segment the {objects[0]} in this image"
        elif len(objects) <= 3:
            prompt = f"Detect and segment {', '.join(objects)} in this {spatial_context} scene"
        else:
            prompt = f"Segment all visible objects, focusing on {', '.join(objects[:3])} and other elements"
        
        if visual_features:
            prompt += f" with {', '.join(visual_features)} characteristics"
        
        return prompt
    
    def _generate_with_clip_fallback(self, 
                                   image: Image.Image,
                                   num_prompts: int) -> List[QwenPromptResult]:
        """Generate prompts using simple fallback."""
        logger.warning("Qwen not available, using simple fallback prompts")
        
        # Generate simple prompts based on image analysis
        simple_prompts = []
        for i in range(num_prompts):
            prompt = f"Analyze this image and identify objects for segmentation. Focus on prominent objects and their spatial relationships."
            qwen_result = QwenPromptResult(
                text=prompt,
                confidence=0.5,
                object_categories=["object"],
                spatial_context="general",
                visual_features=[],
                detailed_description=prompt,
                scene_understanding={'method': 'simple_fallback'},
                reasoning="Simple fallback prompt"
            )
            simple_prompts.append(qwen_result)
        
        return simple_prompts
    
    def generate_specific_prompt(self, 
                               target_object: str, 
                               image: Image.Image,
                               context: str = "") -> str:
        """
        Generate a specific prompt for a target object using Qwen.
        
        Args:
            target_object: The specific object to detect
            image: PIL Image to analyze
            context: Additional context for the prompt
            
        Returns:
            Generated prompt string
        """
        if self.use_qwen:
            image_input = self._prepare_image_for_qwen(image)
            
            specific_prompt = f"""
            Analyze this image and focus specifically on {target_object}. 
            Describe where the {target_object} is located, its appearance, 
            and any relevant details that would help with accurate segmentation.
            {f"Additional context: {context}" if context else ""}
            """
            
            response = self._query_qwen(image_input, specific_prompt)
            parsed = self._parse_qwen_response(response, specific_prompt)
            
            return f"Find and segment {target_object} in this image. {parsed.get('description', '')}"
        else:
            # Fallback to simple prompt
            logger.warning("Qwen not available, using simple fallback prompt")
            return f"Analyze this image and focus specifically on {target_object}. Describe where the {target_object} is located, its appearance, and any relevant details that would help with accurate segmentation."
    
    def analyze_scene_complexity(self, image: Image.Image) -> Dict[str, any]:
        """
        Analyze scene complexity using Qwen's understanding.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            Dictionary with complexity analysis
        """
        if self.use_qwen:
            image_input = self._prepare_image_for_qwen(image)
            
            complexity_prompt = """
            Analyze the complexity of this scene for object detection and segmentation. 
            Consider factors like:
            - Number of objects
            - Object overlap and occlusion
            - Lighting conditions
            - Background complexity
            - Spatial relationships
            - Overall scene difficulty
            
            Provide a complexity score from 1-10 and detailed reasoning.
            """
            
            response = self._query_qwen(image_input, complexity_prompt)
            
            # Parse complexity information
            complexity_score = 5  # Default
            if "complexity score" in response.lower():
                import re
                score_match = re.search(r'(\d+)', response)
                if score_match:
                    complexity_score = int(score_match.group(1))
            
            return {
                'score': complexity_score,
                'analysis': response,
                'difficulty': 'high' if complexity_score > 7 else 'medium' if complexity_score > 4 else 'low',
                'recommended_strategy': self._recommend_strategy(complexity_score)
            }
        else:
            # Fallback analysis
            return {
                'score': 5,
                'analysis': 'CLIP fallback - limited complexity analysis',
                'difficulty': 'medium',
                'recommended_strategy': 'automatic'
            }
    
    def _recommend_strategy(self, complexity_score: int) -> str:
        """Recommend processing strategy based on complexity score."""
        if complexity_score > 8:
            return "guided_high_quality"
        elif complexity_score > 6:
            return "guided_balanced"
        elif complexity_score > 4:
            return "automatic_balanced"
        else:
            return "automatic_fast"
