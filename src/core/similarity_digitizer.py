import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import json
import sqlite3
import os
from datetime import datetime
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class FeatureEmbedding:
    """Container for feature embedding data."""
    feature_id: str
    image_path: str
    bbox: List[int]  # [x1, y1, x2, y2]
    label: str
    confidence: float
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class SimilarityMatch:
    """Container for similarity match results."""
    target_feature: FeatureEmbedding
    similar_features: List[FeatureEmbedding]
    similarity_scores: List[float]
    match_quality: str  # "excellent", "good", "fair", "poor"
    cluster_id: Optional[int] = None

@dataclass
class SimilaritySearchConfig:
    """Configuration for similarity search."""
    similarity_threshold: float = 0.7
    max_results: int = 50
    clustering_enabled: bool = True
    clustering_eps: float = 0.3
    min_cluster_size: int = 2
    use_advanced_metrics: bool = True
    include_metadata: bool = True

class SimilarityDigitizer:
    """
    Advanced similarity digitizer for finding similar features across datasets.
    
    This class provides:
    - Feature embedding extraction and storage
    - Similarity-based search across datasets
    - Clustering of similar features
    - Advanced similarity metrics
    - Visualization of similarity results
    """
    
    def __init__(self, 
                 db_path: str = "similarity_database.db",
                 embedding_model: Optional[Any] = None):
        """
        Initialize the similarity digitizer.
        
        Args:
            db_path: Path to SQLite database for storing embeddings
            embedding_model: Optional pre-trained embedding model
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.feature_embeddings: Dict[str, FeatureEmbedding] = {}
        self.similarity_cache: Dict[str, Dict[str, float]] = {}
        
        # Initialize database
        self._init_database()
        
        # Load existing embeddings
        self._load_embeddings()
        
        logger.info(f"Initialized Similarity Digitizer with {len(self.feature_embeddings)} features")
    
    def _init_database(self):
        """Initialize the similarity database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Features table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                feature_id TEXT PRIMARY KEY,
                image_path TEXT NOT NULL,
                bbox_x1 INTEGER NOT NULL,
                bbox_y1 INTEGER NOT NULL,
                bbox_x2 INTEGER NOT NULL,
                bbox_y2 INTEGER NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT,
                timestamp TEXT NOT NULL
            )
        ''')
        
        # Similarity cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS similarity_cache (
                feature_id_1 TEXT NOT NULL,
                feature_id_2 TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                metric_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                PRIMARY KEY (feature_id_1, feature_id_2, metric_type)
            )
        ''')
        
        # Clusters table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id INTEGER PRIMARY KEY,
                feature_ids TEXT NOT NULL,
                cluster_center BLOB NOT NULL,
                cluster_quality REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_embeddings(self):
        """Load existing embeddings from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT feature_id, image_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                   label, confidence, embedding, metadata, timestamp
            FROM features
        ''')
        
        for row in cursor.fetchall():
            feature_id, image_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2, label, confidence, embedding_blob, metadata_str, timestamp_str = row
            
            # Deserialize embedding
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            
            # Deserialize metadata
            metadata = json.loads(metadata_str) if metadata_str else {}
            
            feature = FeatureEmbedding(
                feature_id=feature_id,
                image_path=image_path,
                bbox=[bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                label=label,
                confidence=confidence,
                embedding=embedding,
                metadata=metadata,
                timestamp=datetime.fromisoformat(timestamp_str)
            )
            
            self.feature_embeddings[feature_id] = feature
        
        conn.close()
        logger.info(f"Loaded {len(self.feature_embeddings)} feature embeddings")
    
    def add_feature(self, 
                   image: Image.Image,
                   bbox: List[int],
                   label: str,
                   confidence: float,
                   embedding: np.ndarray,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new feature to the similarity database.
        
        Args:
            image: PIL Image containing the feature
            bbox: Bounding box [x1, y1, x2, y2]
            label: Feature label/class
            confidence: Detection confidence
            embedding: Feature embedding vector
            metadata: Optional metadata
            
        Returns:
            feature_id: Unique identifier for the feature
        """
        # Generate unique feature ID
        feature_id = self._generate_feature_id(image, bbox, label, embedding)
        
        # Create feature embedding
        feature = FeatureEmbedding(
            feature_id=feature_id,
            image_path=image.filename if hasattr(image, 'filename') else "unknown",
            bbox=bbox,
            label=label,
            confidence=confidence,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        # Store in memory
        self.feature_embeddings[feature_id] = feature
        
        # Store in database
        self._store_feature_in_db(feature)
        
        logger.info(f"Added feature {feature_id} with label '{label}'")
        return feature_id
    
    def _generate_feature_id(self, 
                           image: Image.Image, 
                           bbox: List[int], 
                           label: str, 
                           embedding: np.ndarray) -> str:
        """Generate unique feature ID."""
        image_hash = hashlib.md5(image.tobytes()).hexdigest()[:8]
        bbox_str = "_".join(map(str, bbox))
        embedding_hash = hashlib.md5(embedding.tobytes()).hexdigest()[:8]
        
        return f"{label}_{image_hash}_{bbox_str}_{embedding_hash}"
    
    def _store_feature_in_db(self, feature: FeatureEmbedding):
        """Store feature in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO features 
            (feature_id, image_path, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
             label, confidence, embedding, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feature.feature_id,
            feature.image_path,
            feature.bbox[0], feature.bbox[1], feature.bbox[2], feature.bbox[3],
            feature.label,
            feature.confidence,
            feature.embedding.tobytes(),
            json.dumps(feature.metadata),
            feature.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def find_similar_features(self, 
                            target_feature_id: str,
                            config: Optional[SimilaritySearchConfig] = None) -> SimilarityMatch:
        """
        Find features similar to the target feature.
        
        Args:
            target_feature_id: ID of the target feature
            config: Similarity search configuration
            
        Returns:
            SimilarityMatch with similar features and scores
        """
        if config is None:
            config = SimilaritySearchConfig()
        
        if target_feature_id not in self.feature_embeddings:
            raise ValueError(f"Feature {target_feature_id} not found")
        
        target_feature = self.feature_embeddings[target_feature_id]
        
        # Calculate similarities
        similarities = []
        similar_features = []
        
        for feature_id, feature in self.feature_embeddings.items():
            if feature_id == target_feature_id:
                continue
            
            # Calculate similarity score
            similarity = self._calculate_similarity(
                target_feature.embedding, 
                feature.embedding,
                config.use_advanced_metrics
            )
            
            if similarity >= config.similarity_threshold:
                similarities.append(similarity)
                similar_features.append(feature)
        
        # Sort by similarity score
        sorted_indices = np.argsort(similarities)[::-1]
        similarities = [similarities[i] for i in sorted_indices]
        similar_features = [similar_features[i] for i in sorted_indices]
        
        # Limit results
        if config.max_results:
            similarities = similarities[:config.max_results]
            similar_features = similar_features[:config.max_results]
        
        # Determine match quality
        match_quality = self._determine_match_quality(similarities)
        
        # Perform clustering if enabled
        cluster_id = None
        if config.clustering_enabled and len(similar_features) > 1:
            cluster_id = self._perform_clustering(similar_features, similarities, config)
        
        return SimilarityMatch(
            target_feature=target_feature,
            similar_features=similar_features,
            similarity_scores=similarities,
            match_quality=match_quality,
            cluster_id=cluster_id
        )
    
    def _calculate_similarity(self, 
                            embedding1: np.ndarray, 
                            embedding2: np.ndarray,
                            use_advanced: bool = True) -> float:
        """Calculate similarity between two embeddings."""
        if use_advanced:
            # Cosine similarity
            cos_sim = cosine_similarity([embedding1], [embedding2])[0][0]
            
            # Euclidean distance (normalized)
            euclidean_dist = np.linalg.norm(embedding1 - embedding2)
            euclidean_sim = 1.0 / (1.0 + euclidean_dist)
            
            # Manhattan distance (normalized)
            manhattan_dist = np.sum(np.abs(embedding1 - embedding2))
            manhattan_sim = 1.0 / (1.0 + manhattan_dist)
            
            # Weighted combination
            similarity = 0.6 * cos_sim + 0.3 * euclidean_sim + 0.1 * manhattan_sim
            
        else:
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        
        return float(similarity)
    
    def _determine_match_quality(self, similarities: List[float]) -> str:
        """Determine the quality of similarity matches."""
        if not similarities:
            return "poor"
        
        avg_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)
        
        if avg_similarity >= 0.9 and max_similarity >= 0.95:
            return "excellent"
        elif avg_similarity >= 0.8 and max_similarity >= 0.9:
            return "good"
        elif avg_similarity >= 0.7 and max_similarity >= 0.8:
            return "fair"
        else:
            return "poor"
    
    def _perform_clustering(self, 
                          features: List[FeatureEmbedding],
                          similarities: List[float],
                          config: SimilaritySearchConfig) -> int:
        """Perform clustering on similar features."""
        if len(features) < config.min_cluster_size:
            return None
        
        embeddings = np.array([f.embedding for f in features])
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(
            eps=config.clustering_eps,
            min_samples=config.min_cluster_size,
            metric='cosine'
        ).fit(embeddings)
        
        cluster_labels = clustering.labels_
        if len(set(cluster_labels)) <= 1:  # No meaningful clusters
            return None
        
        # Find largest cluster
        cluster_sizes = {}
        for label in cluster_labels:
            if label != -1:  # Ignore noise
                cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
        
        if not cluster_sizes:
            return None
        
        largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
        
        # Store cluster information
        cluster_id = self._store_cluster(features, cluster_labels, largest_cluster)
        
        return cluster_id
    
    def _store_cluster(self, 
                      features: List[FeatureEmbedding],
                      cluster_labels: np.ndarray,
                      cluster_id: int) -> int:
        """Store cluster information in database."""
        # Get features in the cluster
        cluster_features = [f for i, f in enumerate(features) if cluster_labels[i] == cluster_id]
        
        # Calculate cluster center
        cluster_embeddings = np.array([f.embedding for f in cluster_features])
        cluster_center = np.mean(cluster_embeddings, axis=0)
        
        # Calculate cluster quality (average similarity to center)
        similarities_to_center = [
            cosine_similarity([f.embedding], [cluster_center])[0][0] 
            for f in cluster_features
        ]
        cluster_quality = np.mean(similarities_to_center)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        feature_ids = [f.feature_id for f in cluster_features]
        
        cursor.execute('''
            INSERT OR REPLACE INTO clusters 
            (cluster_id, feature_ids, cluster_center, cluster_quality, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            cluster_id,
            json.dumps(feature_ids),
            cluster_center.tobytes(),
            cluster_quality,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return cluster_id
    
    def search_by_class(self, 
                       class_label: str,
                       config: Optional[SimilaritySearchConfig] = None) -> List[FeatureEmbedding]:
        """
        Find all features belonging to a specific class.
        
        Args:
            class_label: Class label to search for
            config: Search configuration
            
        Returns:
            List of features belonging to the class
        """
        if config is None:
            config = SimilaritySearchConfig()
        
        class_features = [
            feature for feature in self.feature_embeddings.values()
            if feature.label.lower() == class_label.lower()
        ]
        
        # Sort by confidence
        class_features.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit results
        if config.max_results:
            class_features = class_features[:config.max_results]
        
        return class_features
    
    def advanced_similarity_search(self, 
                                 query_embedding: np.ndarray,
                                 class_filter: Optional[str] = None,
                                 config: Optional[SimilaritySearchConfig] = None) -> List[Tuple[FeatureEmbedding, float]]:
        """
        Advanced similarity search using a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            class_filter: Optional class filter
            config: Search configuration
            
        Returns:
            List of (feature, similarity_score) tuples
        """
        if config is None:
            config = SimilaritySearchConfig()
        
        results = []
        
        for feature in self.feature_embeddings.values():
            # Apply class filter if specified
            if class_filter and feature.label.lower() != class_filter.lower():
                continue
            
            # Calculate similarity
            similarity = self._calculate_similarity(
                query_embedding, 
                feature.embedding,
                config.use_advanced_metrics
            )
            
            if similarity >= config.similarity_threshold:
                results.append((feature, similarity))
        
        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Limit results
        if config.max_results:
            results = results[:config.max_results]
        
        return results
    
    def visualize_similarity_results(self, 
                                   similarity_match: SimilarityMatch,
                                   save_path: Optional[str] = None) -> str:
        """
        Create visualization of similarity results.
        
        Args:
            similarity_match: SimilarityMatch result
            save_path: Optional path to save visualization
            
        Returns:
            Path to saved visualization
        """
        if not similarity_match.similar_features:
            logger.warning("No similar features to visualize")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Similarity Analysis: {similarity_match.target_feature.label}', fontsize=16)
        
        # 1. Similarity scores bar chart
        axes[0, 0].bar(range(len(similarity_match.similarity_scores)), 
                      similarity_match.similarity_scores)
        axes[0, 0].set_title('Similarity Scores')
        axes[0, 0].set_xlabel('Feature Index')
        axes[0, 0].set_ylabel('Similarity Score')
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Similarity distribution histogram
        axes[0, 1].hist(similarity_match.similarity_scores, bins=20, alpha=0.7)
        axes[0, 1].set_title('Similarity Score Distribution')
        axes[0, 1].set_xlabel('Similarity Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Confidence vs Similarity scatter plot
        confidences = [f.confidence for f in similarity_match.similar_features]
        axes[1, 0].scatter(confidences, similarity_match.similarity_scores, alpha=0.7)
        axes[1, 0].set_title('Confidence vs Similarity')
        axes[1, 0].set_xlabel('Detection Confidence')
        axes[1, 0].set_ylabel('Similarity Score')
        
        # 4. Feature labels distribution
        labels = [f.label for f in similarity_match.similar_features]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        axes[1, 1].pie(label_counts.values(), labels=label_counts.keys(), autopct='%1.1f%%')
        axes[1, 1].set_title('Feature Label Distribution')
        
        plt.tight_layout()
        
        # Save visualization
        if save_path is None:
            save_path = f"similarity_analysis_{similarity_match.target_feature.feature_id}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Similarity visualization saved to {save_path}")
        return save_path
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        total_features = len(self.feature_embeddings)
        
        # Count by class
        class_counts = {}
        for feature in self.feature_embeddings.values():
            class_counts[feature.label] = class_counts.get(feature.label, 0) + 1
        
        # Average confidence by class
        class_confidences = {}
        for feature in self.feature_embeddings.values():
            if feature.label not in class_confidences:
                class_confidences[feature.label] = []
            class_confidences[feature.label].append(feature.confidence)
        
        avg_confidences = {
            label: np.mean(confidences) 
            for label, confidences in class_confidences.items()
        }
        
        # Database size
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        
        return {
            'total_features': total_features,
            'class_counts': class_counts,
            'avg_confidences': avg_confidences,
            'database_size_mb': db_size / (1024 * 1024),
            'embedding_dimension': len(next(iter(self.feature_embeddings.values())).embedding) if self.feature_embeddings else 0
        }
    
    def export_features(self, 
                       output_path: str,
                       class_filter: Optional[str] = None,
                       format: str = "json") -> str:
        """
        Export features to file.
        
        Args:
            output_path: Output file path
            class_filter: Optional class filter
            format: Export format ("json", "csv")
            
        Returns:
            Path to exported file
        """
        features_to_export = []
        for feature in self.feature_embeddings.values():
            if class_filter and feature.label.lower() != class_filter.lower():
                continue
            features_to_export.append(feature)
        
        if format == "json":
            # Export as JSON
            export_data = []
            for feature in features_to_export:
                export_data.append({
                    'feature_id': feature.feature_id,
                    'image_path': feature.image_path,
                    'bbox': feature.bbox,
                    'label': feature.label,
                    'confidence': feature.confidence,
                    'embedding': feature.embedding.tolist(),
                    'metadata': feature.metadata,
                    'timestamp': feature.timestamp.isoformat()
                })
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        
        elif format == "csv":
            # Export as CSV
            import pandas as pd
            
            data = []
            for feature in features_to_export:
                data.append({
                    'feature_id': feature.feature_id,
                    'image_path': feature.image_path,
                    'bbox_x1': feature.bbox[0],
                    'bbox_y1': feature.bbox[1],
                    'bbox_x2': feature.bbox[2],
                    'bbox_y2': feature.bbox[3],
                    'label': feature.label,
                    'confidence': feature.confidence,
                    'embedding_dim': len(feature.embedding),
                    'timestamp': feature.timestamp.isoformat()
                })
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(features_to_export)} features to {output_path}")
        return output_path
