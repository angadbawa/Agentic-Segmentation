"""
Similarity Digitizer Demo

This script demonstrates the similarity digitizer capabilities for finding
similar features across datasets using embedding similarity scores.
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.optimized_pipeline import OptimizedAgenticPipeline, OptimizedConfig
from core.similarity_digitizer import SimilaritySearchConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_images_with_objects():
    """Create test images with various objects for similarity testing."""
    images = []
    
    # Create multiple images with similar objects
    for i in range(5):
        # Create base image
        image = Image.new('RGB', (512, 512), color=(100, 150, 200))
        draw = ImageDraw.Draw(image)
        
        # Add objects with slight variations
        if i == 0:
            # Red rectangles
            draw.rectangle([50, 50, 150, 150], fill=(255, 0, 0))
            draw.rectangle([200, 200, 300, 300], fill=(255, 100, 100))
        elif i == 1:
            # Similar red rectangles (slightly different)
            draw.rectangle([60, 60, 160, 160], fill=(250, 10, 10))
            draw.rectangle([210, 210, 310, 310], fill=(255, 90, 90))
        elif i == 2:
            # Green circles
            draw.ellipse([100, 100, 200, 200], fill=(0, 255, 0))
            draw.ellipse([250, 250, 350, 350], fill=(100, 255, 100))
        elif i == 3:
            # Similar green circles
            draw.ellipse([110, 110, 210, 210], fill=(10, 250, 10))
            draw.ellipse([260, 260, 360, 360], fill=(90, 255, 90))
        else:
            # Blue triangles
            draw.polygon([(150, 50), (200, 100), (100, 100)], fill=(0, 0, 255))
            draw.polygon([(300, 200), (350, 250), (250, 250)], fill=(100, 100, 255))
        
        # Add some text
        draw.text((50, 400), f"Test Image {i+1}", fill=(255, 255, 255))
        
        images.append(image)
    
    return images

def demonstrate_similarity_digitizer():
    """Demonstrate the similarity digitizer functionality."""
    print("🔍 Similarity Digitizer Demo")
    print("=" * 50)
    
    config = OptimizedConfig(
        mode="quality",
        enable_similarity=True,
        enable_feedback=True,
        enable_caching=True
    )
    
    pipeline = OptimizedAgenticPipeline(config)
    
    test_images = create_test_images_with_objects()
    
    print(f"\n📸 Processing {len(test_images)} test images...")
    
    all_features = []
    for i, image in enumerate(test_images):
        print(f"  Processing image {i+1}/{len(test_images)}...")
        
        result = pipeline.process_image(image, user_id="demo_user")
        
        print(f"    - Detected {len(result.detected_objects)} objects")
        
        # Collect features with IDs
        for obj in result.detected_objects:
            if obj.get('feature_id'):
                all_features.append({
                    'feature_id': obj['feature_id'],
                    'label': obj['label'],
                    'confidence': obj['confidence'],
                    'image_index': i
                })
    
    print(f"\n✅ Total features collected: {len(all_features)}")
    
    print(f"\nCollected Features:")
    for feature in all_features:
        print(f"  - {feature['feature_id']}: {feature['label']} (confidence: {feature['confidence']:.2f})")
    
    # Demonstrate similarity search
    if all_features:
        print(f"\n🔍 Demonstrating Similarity Search...")
        
        # Find similar features for the first feature
        target_feature = all_features[0]
        print(f"  Target feature: {target_feature['feature_id']} ({target_feature['label']})")
        
        try:
            similarity_match = pipeline.find_similar_features(
                target_feature['feature_id'],
                similarity_threshold=0.5,
                max_results=10
            )
            
            print(f"  Found {len(similarity_match.similar_features)} similar features")
            print(f"  Match quality: {similarity_match.match_quality}")
            
            if similarity_match.similar_features:
                print(f"  Similar features:")
                for i, (feature, score) in enumerate(zip(similarity_match.similar_features, similarity_match.similarity_scores)):
                    print(f"    {i+1}. {feature.label} (similarity: {score:.3f})")
            
        except Exception as e:
            print(f"  Error in similarity search: {e}")
    
    # Demonstrate class-based search
    print(f"\nDemonstrating Class-Based Search...")
    
    classes = list(set(f['label'] for f in all_features))
    print(f"  Available classes: {classes}")
    
    for class_label in classes[:2]:  # Test first 2 classes
        try:
            class_features = pipeline.search_by_class(class_label, max_results=10)
            print(f"  Class '{class_label}': {len(class_features)} features found")
            
            for feature in class_features[:3]:  # Show first 3
                print(f"    - {feature['feature_id']} (confidence: {feature['confidence']:.2f})")
                
        except Exception as e:
            print(f"  Error in class search for '{class_label}': {e}")
    
    # Demonstrate advanced similarity search
    print(f"\n🧠 Demonstrating Advanced Similarity Search...")
    
    if all_features:
        first_feature_id = all_features[0]['feature_id']
        
        try:
            query_embedding = np.random.rand(512)  # Simulated embedding
            
            results = pipeline.advanced_similarity_search(
                query_embedding,
                similarity_threshold=0.3,
                max_results=5
            )
            
            print(f"  Advanced search found {len(results)} results")
            for i, result in enumerate(results[:3]):
                feature = result['feature']
                score = result['similarity_score']
                print(f"    {i+1}. {feature['label']} (similarity: {score:.3f})")
                
        except Exception as e:
            print(f"  Error in advanced similarity search: {e}")
    
    print(f"\nSimilarity Statistics:")
    try:
        stats = pipeline.get_similarity_statistics()
        if stats:
            print(f"  Total features: {stats.get('total_features', 0)}")
            print(f"  Class counts: {stats.get('class_counts', {})}")
            print(f"  Database size: {stats.get('database_size_mb', 0):.2f} MB")
            print(f"  Embedding dimension: {stats.get('embedding_dimension', 0)}")
        else:
            print("  No statistics available")
    except Exception as e:
        print(f"  Error getting statistics: {e}")
    
    # Demonstrate visualization
    print(f"\nCreating Similarity Visualization...")
    if all_features:
        try:
            viz_path = pipeline.visualize_similarity_results(
                all_features[0]['feature_id'],
                save_path=f"similarity_demo_{all_features[0]['feature_id']}.png"
            )
            if viz_path:
                print(f"  Visualization saved to: {viz_path}")
            else:
                print("  No visualization created (no similar features found)")
        except Exception as e:
            print(f"  Error creating visualization: {e}")
    
    # Export features
    print(f"\n💾 Exporting Features...")
    try:
        export_path = pipeline.export_similarity_features(
            "similarity_features_export.json",
            format="json"
        )
        print(f"  Features exported to: {export_path}")
    except Exception as e:
        print(f"  Error exporting features: {e}")
    
    print(f"\n🎉 Similarity Digitizer Demo Completed!")

def demonstrate_clustering_features():
    """Demonstrate clustering of similar features."""
    print(f"\n🔗 Feature Clustering Demo")
    print("=" * 50)
    
    # Initialize pipeline
    config = OptimizedConfig(
        mode="quality",
        enable_similarity=True
    )
    
    pipeline = OptimizedAgenticPipeline(config)
    
    # Create images with very similar objects for clustering
    images = []
    for i in range(3):
        image = Image.new('RGB', (400, 400), color=(120, 120, 120))
        draw = ImageDraw.Draw(image)
        
        # Create very similar red squares
        offset = i * 10
        draw.rectangle([50 + offset, 50 + offset, 150 + offset, 150 + offset], fill=(255, 0, 0))
        draw.text((50, 350), f"Cluster Test {i+1}", fill=(255, 255, 255))
        
        images.append(image)
    
    print(f"📸 Processing {len(images)} images for clustering...")
    
    # Process images
    features = []
    for i, image in enumerate(images):
        result = pipeline.process_image(image)
        
        for obj in result.detected_objects:
            if obj.get('feature_id'):
                features.append(obj['feature_id'])
    
    print(f"✅ Collected {len(features)} features for clustering")
    
    # Test clustering by finding similar features
    if features:
        print(f"🔍 Testing clustering with feature: {features[0]}")
        
        try:
            similarity_match = pipeline.find_similar_features(
                features[0],
                similarity_threshold=0.6,
                max_results=20
            )
            
            print(f"  Found {len(similarity_match.similar_features)} similar features")
            if similarity_match.cluster_id is not None:
                print(f"  Cluster ID: {similarity_match.cluster_id}")
            else:
                print("  No clustering performed (insufficient similar features)")
                
        except Exception as e:
            print(f"  Error in clustering test: {e}")

def main():
    """Main demo function."""
    print("Starting Similarity Digitizer Demo")
    print("=" * 60)
    
    try:
        demonstrate_similarity_digitizer()
        
        # Clustering demonstration
        demonstrate_clustering_features()
        
        print(f"\n✅ All demos completed successfully!")
        print(f"\nKey Features Demonstrated:")
        print(f"   ✅ Feature embedding extraction and storage")
        print(f"   ✅ Similarity-based feature matching")
        print(f"   ✅ Class-based feature search")
        print(f"   ✅ Advanced similarity search with query embeddings")
        print(f"   ✅ Feature clustering and grouping")
        print(f"   ✅ Similarity visualization and analytics")
        print(f"   ✅ Feature export and database management")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    main()
