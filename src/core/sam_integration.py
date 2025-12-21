import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path

# SAM imports
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    from segment_anything.utils.transforms import ResizeLongestSide
except ImportError:
    logger.warning("SAM not installed. Please install with: pip install segment-anything")
    sam_model_registry = None
    SamPredictor = None
    SamAutomaticMaskGenerator = None

logger = logging.getLogger(__name__)

@dataclass
class SegmentationResult:
    """Container for segmentation results."""
    masks: np.ndarray  # Binary masks
    scores: np.ndarray  # Confidence scores
    labels: List[str]  # Object labels
    boxes: np.ndarray  # Bounding boxes
    metadata: Dict[str, any]

@dataclass
class DetectionResult:
    """Container for detection results."""
    boxes: np.ndarray  # Bounding boxes [x1, y1, x2, y2]
    scores: np.ndarray  # Confidence scores
    labels: List[str]  # Object labels
    masks: Optional[np.ndarray] = None  # Optional segmentation masks

class SAMSegmentation:
    """
    Comprehensive SAM integration for object detection and segmentation.
    Supports both automatic and prompt-based segmentation.
    """
    
    def __init__(self, 
                 model_type: str = "vit_b",
                 checkpoint_path: Optional[str] = None,
                 device: str = "auto"):
        """
        Initialize SAM segmentation.
        
        Args:
            model_type: SAM model type ("vit_b", "vit_l", "vit_h")
            checkpoint_path: Path to SAM checkpoint file
            device: Device to run SAM on
        """
        if sam_model_registry is None:
            raise ImportError("SAM not installed. Please install with: pip install segment-anything")
        
        self.device = self._get_device(device)
        self.model_type = model_type
        
        self.sam = self._load_sam_model(checkpoint_path)
        self.predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        
        logger.info(f"Initialized SAM with {model_type} on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_sam_model(self, checkpoint_path: Optional[str]) -> torch.nn.Module:
        """Load SAM model from checkpoint."""
        if checkpoint_path is None:
            checkpoint_paths = {
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            }
            checkpoint_path = checkpoint_paths.get(self.model_type)
        
        if checkpoint_path.startswith("http"):
            # Download checkpoint if needed
            checkpoint_path = self._download_checkpoint(checkpoint_path)
        
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        return sam
    
    def _download_checkpoint(self, url: str) -> str:
        """Download SAM checkpoint if not present."""
        import requests
        from pathlib import Path
        
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        filename = url.split("/")[-1]
        checkpoint_path = checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            logger.info(f"Downloading SAM checkpoint: {filename}")
            response = requests.get(url)
            response.raise_for_status()
            
            with open(checkpoint_path, 'wb') as f:
                f.write(response.content)
        
        return str(checkpoint_path)
    
    def segment_with_prompts(self, 
                           image: Image.Image,
                           prompts: Dict[str, Union[List, np.ndarray]],
                           labels: List[str] = None) -> SegmentationResult:
        """
        Segment objects using various types of prompts.
        
        Args:
            image: PIL Image to segment
            prompts: Dictionary with prompt types:
                - 'points': List of (x, y) coordinates
                - 'boxes': List of [x1, y1, x2, y2] bounding boxes
                - 'masks': Previous masks as guidance
            labels: Optional labels for prompts
            
        Returns:
            SegmentationResult with masks and metadata
        """
        try:
            # Convert PIL to numpy
            image_array = np.array(image)
            
            # Set image for predictor
            self.predictor.set_image(image_array)
            
            # Prepare input prompts
            input_points = None
            input_labels = None
            input_boxes = None
            input_masks = None
            
            if 'points' in prompts:
                input_points = np.array(prompts['points'])
                input_labels = np.array([1] * len(input_points))  # Default to foreground
            
            if 'boxes' in prompts:
                input_boxes = np.array(prompts['boxes'])
            
            if 'masks' in prompts:
                input_masks = np.array(prompts['masks'])
            
            # Run prediction
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_boxes[0] if input_boxes is not None and len(input_boxes) > 0 else None,
                mask_input=input_masks[0] if input_masks is not None and len(input_masks) > 0 else None,
                multimask_output=True
            )
            
            # Use the best mask
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            best_score = scores[best_mask_idx]
            
            # Convert to binary mask
            binary_mask = (best_mask > 0.5).astype(np.uint8)
            
            # Generate bounding box
            bbox = self._mask_to_bbox(binary_mask)
            
            # Create result
            result = SegmentationResult(
                masks=binary_mask,
                scores=np.array([best_score]),
                labels=labels or ["object"],
                boxes=np.array([bbox]),
                metadata={
                    'prompt_type': list(prompts.keys()),
                    'num_prompts': len(prompts.get('points', [])) + len(prompts.get('boxes', [])),
                    'model_type': self.model_type
                }
            )
            
            logger.info(f"Segmented object with score {best_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in prompt-based segmentation: {e}")
            return self._get_empty_segmentation_result()
    
    def segment_automatically(self, 
                            image: Image.Image,
                            min_mask_region_area: int = 100,
                            max_detections: int = 100) -> SegmentationResult:
        """
        Automatically segment all objects in an image.
        
        Args:
            image: PIL Image to segment
            min_mask_region_area: Minimum area for mask regions
            max_detections: Maximum number of detections
            
        Returns:
            SegmentationResult with all detected objects
        """
        try:
            # Convert PIL to numpy
            image_array = np.array(image)
            
            # Configure mask generator
            self.mask_generator.min_mask_region_area = min_mask_region_area
            
            # Generate masks
            masks = self.mask_generator.generate(image_array)
            
            # Limit number of detections
            if len(masks) > max_detections:
                # Sort by stability score and take top ones
                masks = sorted(masks, key=lambda x: x['stability_score'], reverse=True)[:max_detections]
            
            # Extract data
            binary_masks = np.array([mask['segmentation'] for mask in masks])
            scores = np.array([mask['stability_score'] for mask in masks])
            bboxes = np.array([mask['bbox'] for mask in masks])  # [x, y, w, h]
            
            # Convert bboxes to [x1, y1, x2, y2] format
            boxes = np.array([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] 
                            for bbox in bboxes])
            
            # Generate labels
            labels = [f"object_{i+1}" for i in range(len(masks))]
            
            result = SegmentationResult(
                masks=binary_masks,
                scores=scores,
                labels=labels,
                boxes=boxes,
                metadata={
                    'method': 'automatic',
                    'num_detections': len(masks),
                    'min_area': min_mask_region_area,
                    'model_type': self.model_type
                }
            )
            
            logger.info(f"Automatically segmented {len(masks)} objects")
            return result
            
        except Exception as e:
            logger.error(f"Error in automatic segmentation: {e}")
            return self._get_empty_segmentation_result()
    
    def segment_with_click(self, 
                          image: Image.Image,
                          click_points: List[Tuple[int, int]],
                          click_labels: List[int] = None) -> SegmentationResult:
        """
        Segment objects using click points.
        
        Args:
            image: PIL Image to segment
            click_points: List of (x, y) click coordinates
            click_labels: List of labels (1 for foreground, 0 for background)
            
        Returns:
            SegmentationResult
        """
        if click_labels is None:
            click_labels = [1] * len(click_points)  # Default to foreground
        
        prompts = {
            'points': click_points,
            'labels': click_labels
        }
        
        return self.segment_with_prompts(image, prompts)
    
    def segment_with_bbox(self, 
                         image: Image.Image,
                         bbox: List[int]) -> SegmentationResult:
        """
        Segment object using bounding box.
        
        Args:
            image: PIL Image to segment
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            SegmentationResult
        """
        prompts = {
            'boxes': [bbox]
        }
        
        return self.segment_with_prompts(image, prompts)
    
    def refine_segmentation(self, 
                           image: Image.Image,
                           previous_mask: np.ndarray,
                           refinement_points: List[Tuple[int, int]],
                           refinement_labels: List[int]) -> SegmentationResult:
        """
        Refine existing segmentation with additional points.
        
        Args:
            image: PIL Image
            previous_mask: Previous segmentation mask
            refinement_points: Points for refinement
            refinement_labels: Labels for refinement points
            
        Returns:
            Refined SegmentationResult
        """
        prompts = {
            'masks': [previous_mask],
            'points': refinement_points,
            'labels': refinement_labels
        }
        
        return self.segment_with_prompts(image, prompts)
    
    def detect_objects(self, 
                      image: Image.Image,
                      confidence_threshold: float = 0.5) -> DetectionResult:
        """
        Detect objects in image and return bounding boxes.
        
        Args:
            image: PIL Image to process
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            DetectionResult with bounding boxes and scores
        """
        # Use automatic segmentation to get objects
        seg_result = self.segment_automatically(image)
        
        # Filter by confidence
        valid_indices = seg_result.scores >= confidence_threshold
        filtered_boxes = seg_result.boxes[valid_indices]
        filtered_scores = seg_result.scores[valid_indices]
        filtered_labels = [seg_result.labels[i] for i in range(len(seg_result.labels)) if valid_indices[i]]
        filtered_masks = seg_result.masks[valid_indices] if len(seg_result.masks.shape) > 2 else seg_result.masks
        
        return DetectionResult(
            boxes=filtered_boxes,
            scores=filtered_scores,
            labels=filtered_labels,
            masks=filtered_masks
        )
    
    def _mask_to_bbox(self, mask: np.ndarray) -> List[int]:
        """Convert binary mask to bounding box."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return [0, 0, 0, 0]
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return [x_min, y_min, x_max, y_max]
    
    def _get_empty_segmentation_result(self) -> SegmentationResult:
        """Get empty segmentation result for error cases."""
        return SegmentationResult(
            masks=np.zeros((1, 1, 1), dtype=np.uint8),
            scores=np.array([0.0]),
            labels=["unknown"],
            boxes=np.array([[0, 0, 0, 0]]),
            metadata={'error': True}
        )
    
    def visualize_results(self, 
                         image: Image.Image,
                         result: SegmentationResult,
                         alpha: float = 0.5) -> Image.Image:
        """
        Visualize segmentation results on the image.
        
        Args:
            image: Original PIL Image
            result: SegmentationResult to visualize
            alpha: Transparency for overlay
            
        Returns:
            PIL Image with visualization
        """
        try:
            # Convert to numpy
            img_array = np.array(image)
            vis_image = img_array.copy()
            
            # Create color map for different objects
            colors = self._generate_colors(len(result.labels))
            
            # Overlay masks
            for i, (mask, color) in enumerate(zip(result.masks, colors)):
                if len(mask.shape) == 2:
                    # Single mask
                    colored_mask = np.zeros_like(img_array)
                    colored_mask[mask > 0] = color
                    vis_image = cv2.addWeighted(vis_image, 1-alpha, colored_mask, alpha, 0)
                else:
                    # Multiple masks
                    for j, single_mask in enumerate(mask):
                        colored_mask = np.zeros_like(img_array)
                        colored_mask[single_mask > 0] = colors[j % len(colors)]
                        vis_image = cv2.addWeighted(vis_image, 1-alpha, colored_mask, alpha, 0)
            
            # Draw bounding boxes
            for i, (box, label, score) in enumerate(zip(result.boxes, result.labels, result.scores)):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), colors[i % len(colors)], 2)
                
                # Add label and score
                label_text = f"{label}: {score:.2f}"
                cv2.putText(vis_image, label_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 1)
            
            return Image.fromarray(vis_image)
            
        except Exception as e:
            logger.error(f"Error visualizing results: {e}")
            return image
    
    def _generate_colors(self, num_colors: int) -> List[List[int]]:
        """Generate distinct colors for visualization."""
        import colorsys
        
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
            colors.append([int(c * 255) for c in rgb])
        
        return colors
