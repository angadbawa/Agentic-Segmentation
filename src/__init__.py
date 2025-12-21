"""
Zero-Context Agentic Object Detection & Segmentation Pipeline

A sophisticated AI-powered pipeline for automatic object detection and segmentation
without requiring manual context or prompts.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .core.optimized_pipeline import OptimizedAgenticPipeline, OptimizedConfig, OptimizedResult
from .core.feedback_system import HumanFeedbackSystem, FeedbackType, FeedbackAction
from .core.similarity_digitizer import SimilarityDigitizer, SimilaritySearchConfig, SimilarityMatch

__all__ = [
    "OptimizedAgenticPipeline",
    "OptimizedConfig",
    "OptimizedResult",
    "HumanFeedbackSystem",
    "FeedbackType",
    "FeedbackAction",
    "SimilarityDigitizer",
    "SimilaritySearchConfig",
    "SimilarityMatch"
]
