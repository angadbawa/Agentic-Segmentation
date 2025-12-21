"""
Human Feedback System Demo

This script demonstrates the human feedback capabilities of the optimized
agentic pipeline, showing how user corrections improve future processing.
"""

import sys
import os
import time
import numpy as np
from PIL import Image, ImageDraw
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.optimized_pipeline import OptimizedAgenticPipeline, OptimizedConfig
from core.feedback_system import FeedbackType, FeedbackAction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image() -> Image.Image:
    """Create a test image with multiple objects."""
    # Create a 512x512 image
    image = Image.new('RGB', (512, 512), color=(100, 150, 200))
    draw = ImageDraw.Draw(image)
    
    # Add some objects
    draw.rectangle([50, 50, 150, 150], fill=(255, 0, 0))      # Red rectangle
    draw.ellipse([200, 200, 300, 300], fill=(0, 255, 0))     # Green circle
    draw.polygon([(400, 100), (450, 50), (500, 100), (450, 150)], fill=(0, 0, 255))  # Blue triangle
    
    # Add some text
    draw.text((50, 200), "Test Image", fill=(255, 255, 255))
    
    return image

def demonstrate_feedback_system():
    """Demonstrate the human feedback system."""
    print("🤖 Human Feedback System Demo")
    print("=" * 50)
    
    config = OptimizedConfig(
        mode="auto",
        enable_feedback=True,
        enable_caching=True
    )
    
    pipeline = OptimizedAgenticPipeline(config)
    
    test_image = create_test_image()
    user_id = "demo_user"
    
    print(f"\nProcessing test image...")
    
    result = pipeline.process_image(test_image, user_id=user_id)
    
    print(f"[OK] Processing completed!")
    print(f"   - Detected {len(result.detected_objects)} objects")
    print(f"   - Processing time: {result.processing_time:.2f}s")
    print(f"   - Mode used: {result.mode_used}")
    print(f"   - Feedback session ID: {result.feedback_session_id}")
    
    print(f"\nDetected Objects:")
    for i, obj in enumerate(result.detected_objects):
        print(f"   {i+1}. {obj['label']} (confidence: {obj['confidence']:.2f})")
    
    # Simulate user feedback
    print(f"\nSimulating user feedback...")
    
    # User corrects object labels and confidence
    object_feedbacks = [
        {
            'object_id': '0',
            'action': 'modify',
            'original_detection': result.detected_objects[0],
            'corrected_detection': {
                **result.detected_objects[0],
                'label': 'red_rectangle',
                'confidence': 0.9
            },
            'user_comment': 'This is clearly a red rectangle',
            'confidence_rating': 0.9
        },
        {
            'object_id': '1',
            'action': 'modify',
            'original_detection': result.detected_objects[1],
            'corrected_detection': {
                **result.detected_objects[1],
                'label': 'green_circle',
                'confidence': 0.85
            },
            'user_comment': 'This is a green circle',
            'confidence_rating': 0.85
        },
        {
            'object_id': '2',
            'action': 'add',
            'original_detection': result.detected_objects[2],
            'corrected_detection': {
                'label': 'blue_triangle',
                'confidence': 0.8,
                'bbox': [400, 50, 500, 150],
                'mask': result.detected_objects[2]['mask']
            },
            'user_comment': 'This is a blue triangle',
            'confidence_rating': 0.8
        }
    ]
    
    # User provides overall processing feedback
    processing_feedback = {
        'feedback_type': 'rating',
        'overall_rating': 0.8,
        'processing_time_acceptable': True,
        'preferred_mode': 'quality',
        'user_comment': 'Good detection, but could be more accurate',
        'suggestions': ['Improve object classification', 'Better confidence scoring']
    }
    
    pipeline.add_feedback(
        feedback_session_id=result.feedback_session_id,
        object_feedbacks=object_feedbacks,
        processing_feedback=processing_feedback
    )
    
    print(f"[OK] Feedback added successfully!")
    
    print(f"\n🔄 Processing same image again with learned feedback...")
    
    result2 = pipeline.process_image(test_image, user_id=user_id)
    
    print(f"[OK] Second processing completed!")
    print(f"   - Detected {len(result2.detected_objects)} objects")
    print(f"   - Processing time: {result2.processing_time:.2f}s")
    print(f"   - Mode used: {result2.mode_used}")
    
    print(f"\nImproved Detected Objects:")
    for i, obj in enumerate(result2.detected_objects):
        print(f"   {i+1}. {obj['label']} (confidence: {obj['confidence']:.2f})")
    
    print(f"\nGetting adaptive suggestions...")
    suggestions = pipeline.get_adaptive_suggestions(test_image, user_id)
    
    print(f"Adaptive Suggestions:")
    print(f"   - Recommended mode: {suggestions.get('recommended_mode', 'auto')}")
    print(f"   - Expected objects: {suggestions.get('expected_objects', [])}")
    print(f"   - Confidence adjustments: {suggestions.get('confidence_adjustments', {})}")
    
    print(f"\nFeedback Statistics:")
    stats = pipeline.get_feedback_statistics(user_id)
    
    if stats:
        print(f"   - Total sessions: {stats.get('total_sessions', 0)}")
        print(f"   - Total object feedbacks: {stats.get('total_object_feedbacks', 0)}")
        print(f"   - Average ratings: {stats.get('avg_ratings', {})}")
        print(f"   - Action distribution: {stats.get('action_distribution', {})}")
    else:
        print(f"   - No feedback statistics available yet")
    
    print(f"\nDemo completed successfully!")

def demonstrate_learning_improvement():
    """Demonstrate how the system learns and improves over time."""
    print(f"\nLearning Improvement Demo")
    print("=" * 50)
    
    # Initialize pipeline
    config = OptimizedConfig(
        mode="auto",
        enable_feedback=True
    )
    
    pipeline = OptimizedAgenticPipeline(config)
    user_id = "learning_user"
    
    # Process multiple images with feedback
    for i in range(3):
        print(f"\nProcessing image {i+1}/3...")
        
        # Create slightly different test image
        test_image = create_test_image()
        if i == 1:
            # Modify image slightly
            draw = ImageDraw.Draw(test_image)
            draw.rectangle([100, 100, 200, 200], fill=(255, 255, 0))  # Yellow rectangle
        
        # Process image
        result = pipeline.process_image(test_image, user_id=user_id)
        
        print(f"   - Detected {len(result.detected_objects)} objects")
        print(f"   - Mode used: {result.mode_used}")
        
        # Simulate feedback (user gets better at providing feedback)
        if result.feedback_session_id:
            object_feedbacks = []
            for j, obj in enumerate(result.detected_objects):
                object_feedbacks.append({
                    'object_id': str(j),
                    'action': 'confirm' if j < 2 else 'modify',
                    'original_detection': obj,
                    'user_comment': f'Object {j+1} feedback',
                    'confidence_rating': min(0.95, obj['confidence'] + 0.1)  # Slightly improve confidence
                })
            
            processing_feedback = {
                'feedback_type': 'rating',
                'overall_rating': 0.7 + (i * 0.1),  # Improving over time
                'processing_time_acceptable': True,
                'preferred_mode': 'quality' if i > 1 else 'auto',
                'user_comment': f'Processing {i+1} feedback'
            }
            
            pipeline.add_feedback(
                feedback_session_id=result.feedback_session_id,
                object_feedbacks=object_feedbacks,
                processing_feedback=processing_feedback
            )
            
            print(f"   - Feedback added (rating: {processing_feedback['overall_rating']:.1f})")
    
    # Show final statistics
    print(f"\nFinal Learning Statistics:")
    stats = pipeline.get_feedback_statistics(user_id)
    
    if stats:
        print(f"   - Total sessions: {stats.get('total_sessions', 0)}")
        print(f"   - Average overall rating: {stats.get('avg_ratings', {}).get('overall', 0):.2f}")
        print(f"   - Feedback types: {stats.get('feedback_types', {})}")
        print(f"   - Action distribution: {stats.get('action_distribution', {})}")
    
    print(f"\nLearning improvement demonstrated!")

def main():
    """Main demo function."""
    print("Starting Human Feedback System Demo")
    print("=" * 60)
    
    try:
        demonstrate_feedback_system()
        
        # Learning improvement demonstration
        demonstrate_learning_improvement()
        
        print(f"\n[OK] All demos completed successfully!")
        print(f"\nKey Features Demonstrated:")
        print(f"   [OK] Human feedback collection")
        print(f"   [OK] Adaptive behavior based on feedback")
        print(f"   [OK] Learning from user corrections")
        print(f"   [OK] Confidence adjustment")
        print(f"   [OK] Mode preference learning")
        print(f"   [OK] Feedback statistics and analytics")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"[ERROR] Demo failed: {e}")

if __name__ == "__main__":
    main()
