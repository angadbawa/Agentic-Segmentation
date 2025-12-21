import json
import time
import hashlib
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of human feedback."""
    CORRECTION = "correction"           # User corrects wrong detection
    CONFIRMATION = "confirmation"       # User confirms correct detection
    REFINEMENT = "refinement"          # User provides refinement hints
    PREFERENCE = "preference"          # User expresses processing preference
    RATING = "rating"                  # User rates overall result quality

class FeedbackAction(Enum):
    """Actions user can take on detected objects."""
    ADD = "add"                        # Add missing object
    REMOVE = "remove"                  # Remove false positive
    MODIFY = "modify"                  # Modify object properties
    CONFIRM = "confirm"                # Confirm object is correct
    REJECT = "reject"                  # Reject object detection

@dataclass
class ObjectFeedback:
    """Feedback for a specific detected object."""
    object_id: str
    action: FeedbackAction
    original_detection: Dict[str, Any]
    corrected_detection: Optional[Dict[str, Any]] = None
    user_comment: Optional[str] = None
    confidence_rating: Optional[float] = None  # 0-1 scale

@dataclass
class ProcessingFeedback:
    """Feedback for overall processing."""
    feedback_type: FeedbackType
    overall_rating: Optional[float] = None  # 0-1 scale
    processing_time_acceptable: Optional[bool] = None
    preferred_mode: Optional[str] = None
    user_comment: Optional[str] = None
    suggestions: Optional[List[str]] = None

@dataclass
class FeedbackSession:
    """Complete feedback session for an image processing."""
    session_id: str
    image_hash: str
    timestamp: datetime
    original_result: Dict[str, Any]
    object_feedbacks: List[ObjectFeedback]
    processing_feedback: Optional[ProcessingFeedback]
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class FeedbackDatabase:
    """Database for storing and retrieving feedback data."""
    
    def __init__(self, db_path: str = "feedback.db"):
        """Initialize feedback database."""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_sessions (
                session_id TEXT PRIMARY KEY,
                image_hash TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                original_result TEXT NOT NULL,
                user_id TEXT,
                context TEXT
            )
        ''')
        
        # Object feedbacks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS object_feedbacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                object_id TEXT NOT NULL,
                action TEXT NOT NULL,
                original_detection TEXT NOT NULL,
                corrected_detection TEXT,
                user_comment TEXT,
                confidence_rating REAL,
                FOREIGN KEY (session_id) REFERENCES feedback_sessions (session_id)
            )
        ''')
        
        # Processing feedbacks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_feedbacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                overall_rating REAL,
                processing_time_acceptable BOOLEAN,
                preferred_mode TEXT,
                user_comment TEXT,
                suggestions TEXT,
                FOREIGN KEY (session_id) REFERENCES feedback_sessions (session_id)
            )
        ''')
        
        # Learning patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                last_used TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_feedback_session(self, session: FeedbackSession):
        """Store a complete feedback session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Store session
            cursor.execute('''
                INSERT OR REPLACE INTO feedback_sessions 
                (session_id, image_hash, timestamp, original_result, user_id, context)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session.session_id,
                session.image_hash,
                session.timestamp.isoformat(),
                json.dumps(session.original_result),
                session.user_id,
                json.dumps(session.context) if session.context else None
            ))
            
            # Store object feedbacks
            for obj_feedback in session.object_feedbacks:
                cursor.execute('''
                    INSERT INTO object_feedbacks 
                    (session_id, object_id, action, original_detection, 
                     corrected_detection, user_comment, confidence_rating)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session.session_id,
                    obj_feedback.object_id,
                    obj_feedback.action.value,
                    json.dumps(obj_feedback.original_detection),
                    json.dumps(obj_feedback.corrected_detection) if obj_feedback.corrected_detection else None,
                    obj_feedback.user_comment,
                    obj_feedback.confidence_rating
                ))
            
            # Store processing feedback
            if session.processing_feedback:
                cursor.execute('''
                    INSERT INTO processing_feedbacks 
                    (session_id, feedback_type, overall_rating, processing_time_acceptable,
                     preferred_mode, user_comment, suggestions)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session.session_id,
                    session.processing_feedback.feedback_type.value,
                    session.processing_feedback.overall_rating,
                    session.processing_feedback.processing_time_acceptable,
                    session.processing_feedback.preferred_mode,
                    session.processing_feedback.user_comment,
                    json.dumps(session.processing_feedback.suggestions) if session.processing_feedback.suggestions else None
                ))
            
            conn.commit()
            logger.info(f"Stored feedback session: {session.session_id}")
            
        except Exception as e:
            logger.error(f"Error storing feedback session: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_feedback_history(self, image_hash: str = None, user_id: str = None, limit: int = 100) -> List[FeedbackSession]:
        """Retrieve feedback history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT session_id, image_hash, timestamp, original_result, user_id, context
            FROM feedback_sessions
        '''
        params = []
        
        if image_hash:
            query += ' WHERE image_hash = ?'
            params.append(image_hash)
        
        if user_id:
            if image_hash:
                query += ' AND user_id = ?'
            else:
                query += ' WHERE user_id = ?'
            params.append(user_id)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        sessions = []
        
        for row in cursor.fetchall():
            session_id, image_hash, timestamp, original_result, user_id, context = row
            
            cursor.execute('''
                SELECT object_id, action, original_detection, corrected_detection,
                       user_comment, confidence_rating
                FROM object_feedbacks WHERE session_id = ?
            ''', (session_id,))
            
            object_feedbacks = []
            for obj_row in cursor.fetchall():
                object_feedbacks.append(ObjectFeedback(
                    object_id=obj_row[0],
                    action=FeedbackAction(obj_row[1]),
                    original_detection=json.loads(obj_row[2]),
                    corrected_detection=json.loads(obj_row[3]) if obj_row[3] else None,
                    user_comment=obj_row[4],
                    confidence_rating=obj_row[5]
                ))
            
            cursor.execute('''
                SELECT feedback_type, overall_rating, processing_time_acceptable,
                       preferred_mode, user_comment, suggestions
                FROM processing_feedbacks WHERE session_id = ?
            ''', (session_id,))
            
            processing_feedback = None
            proc_row = cursor.fetchone()
            if proc_row:
                processing_feedback = ProcessingFeedback(
                    feedback_type=FeedbackType(proc_row[0]),
                    overall_rating=proc_row[1],
                    processing_time_acceptable=proc_row[2],
                    preferred_mode=proc_row[3],
                    user_comment=proc_row[4],
                    suggestions=json.loads(proc_row[5]) if proc_row[5] else None
                )
            
            sessions.append(FeedbackSession(
                session_id=session_id,
                image_hash=image_hash,
                timestamp=datetime.fromisoformat(timestamp),
                original_result=json.loads(original_result),
                object_feedbacks=object_feedbacks,
                processing_feedback=processing_feedback,
                user_id=user_id,
                context=json.loads(context) if context else None
            ))
        
        conn.close()
        return sessions

class HumanFeedbackSystem:
    """
    Comprehensive human feedback system for learning and improvement.
    
    This system implements Chapter 13: Human-in-the-Loop patterns with
    advanced learning capabilities.
    """
    
    def __init__(self, db_path: str = "feedback.db"):
        """Initialize the feedback system."""
        self.database = FeedbackDatabase(db_path)
        self.learning_patterns = {}
        self.user_preferences = {}
        self.performance_history = []
        
        logger.info("Initialized Human Feedback System")
    
    def collect_feedback(self, 
                        image: Image.Image,
                        original_result: Dict[str, Any],
                        user_id: Optional[str] = None,
                        context: Optional[Dict[str, Any]] = None) -> str:
        """
        Collect comprehensive feedback from user.
        
        Args:
            image: Original processed image
            original_result: Result from pipeline processing
            user_id: Optional user identifier
            context: Optional context information
            
        Returns:
            session_id: Unique identifier for this feedback session
        """
        # Generate session ID
        image_hash = self._hash_image(image)
        session_id = f"{image_hash}_{int(time.time())}"
        
        # Create feedback session
        session = FeedbackSession(
            session_id=session_id,
            image_hash=image_hash,
            timestamp=datetime.now(),
            original_result=original_result,
            object_feedbacks=[],  # Will be populated by user
            processing_feedback=None,  # Will be populated by user
            user_id=user_id,
            context=context
        )
        
        # Store initial session
        self.database.store_feedback_session(session)
        
        logger.info(f"Created feedback session: {session_id}")
        return session_id
    
    def add_object_feedback(self, 
                           session_id: str,
                           object_id: str,
                           action: FeedbackAction,
                           original_detection: Dict[str, Any],
                           corrected_detection: Optional[Dict[str, Any]] = None,
                           user_comment: Optional[str] = None,
                           confidence_rating: Optional[float] = None):
        """Add feedback for a specific detected object."""
        feedback = ObjectFeedback(
            object_id=object_id,
            action=action,
            original_detection=original_detection,
            corrected_detection=corrected_detection,
            user_comment=user_comment,
            confidence_rating=confidence_rating
        )
        
        # Retrieve existing session
        sessions = self.database.get_feedback_history(limit=1000)
        session = next((s for s in sessions if s.session_id == session_id), None)
        
        if session:
            session.object_feedbacks.append(feedback)
            self.database.store_feedback_session(session)
            logger.info(f"Added object feedback for {object_id} in session {session_id}")
        else:
            logger.error(f"Session {session_id} not found")
    
    def add_processing_feedback(self,
                               session_id: str,
                               feedback_type: FeedbackType,
                               overall_rating: Optional[float] = None,
                               processing_time_acceptable: Optional[bool] = None,
                               preferred_mode: Optional[str] = None,
                               user_comment: Optional[str] = None,
                               suggestions: Optional[List[str]] = None):
        """Add feedback for overall processing."""
        feedback = ProcessingFeedback(
            feedback_type=feedback_type,
            overall_rating=overall_rating,
            processing_time_acceptable=processing_time_acceptable,
            preferred_mode=preferred_mode,
            user_comment=user_comment,
            suggestions=suggestions
        )
        
        # Retrieve existing session
        sessions = self.database.get_feedback_history(limit=1000)
        session = next((s for s in sessions if s.session_id == session_id), None)
        
        if session:
            session.processing_feedback = feedback
            self.database.store_feedback_session(session)
            logger.info(f"Added processing feedback for session {session_id}")
        else:
            logger.error(f"Session {session_id} not found")
    
    def learn_from_feedback(self, session_id: str) -> Dict[str, Any]:
        """
        Learn from feedback and update system behavior.
        
        This implements Chapter 9: Learning and Adaptation patterns.
        """
        sessions = self.database.get_feedback_history(limit=1000)
        session = next((s for s in sessions if s.session_id == session_id), None)
        
        if not session:
            logger.error(f"Session {session_id} not found for learning")
            return {}
        
        learning_insights = {
            'object_patterns': {},
            'processing_preferences': {},
            'confidence_adjustments': {},
            'mode_preferences': {}
        }
        
        # Learn from object feedbacks
        for obj_feedback in session.object_feedbacks:
            object_type = obj_feedback.original_detection.get('label', 'unknown')
            
            if object_type not in learning_insights['object_patterns']:
                learning_insights['object_patterns'][object_type] = {
                    'corrections': 0,
                    'confirmations': 0,
                    'avg_confidence': 0,
                    'common_issues': []
                }
            
            pattern = learning_insights['object_patterns'][object_type]
            
            if obj_feedback.action == FeedbackAction.CORRECT:
                pattern['confirmations'] += 1
            elif obj_feedback.action in [FeedbackAction.ADD, FeedbackAction.MODIFY, FeedbackAction.REMOVE]:
                pattern['corrections'] += 1
                if obj_feedback.user_comment:
                    pattern['common_issues'].append(obj_feedback.user_comment)
            
            # Update confidence patterns
            if obj_feedback.confidence_rating is not None:
                original_confidence = obj_feedback.original_detection.get('confidence', 0.5)
                confidence_diff = obj_feedback.confidence_rating - original_confidence
                
                if object_type not in learning_insights['confidence_adjustments']:
                    learning_insights['confidence_adjustments'][object_type] = []
                
                learning_insights['confidence_adjustments'][object_type].append(confidence_diff)
        
        # Learn from processing feedback
        if session.processing_feedback:
            proc_feedback = session.processing_feedback
            
            if proc_feedback.preferred_mode:
                learning_insights['mode_preferences'][session.image_hash] = proc_feedback.preferred_mode
            
            if proc_feedback.overall_rating is not None:
                learning_insights['processing_preferences']['avg_rating'] = proc_feedback.overall_rating
            
            if proc_feedback.processing_time_acceptable is not None:
                learning_insights['processing_preferences']['time_acceptable'] = proc_feedback.processing_time_acceptable
        
        # Store learning patterns
        self._store_learning_patterns(learning_insights)
        
        logger.info(f"Learned from feedback session: {session_id}")
        return learning_insights
    
    def get_adaptive_suggestions(self, image: Image.Image, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get adaptive suggestions based on learned patterns.
        
        This implements adaptive behavior based on user feedback history.
        """
        image_hash = self._hash_image(image)
        suggestions = {
            'recommended_mode': 'auto',
            'confidence_adjustments': {},
            'expected_objects': [],
            'processing_hints': []
        }
        
        # Get user's feedback history
        user_sessions = self.database.get_feedback_history(user_id=user_id, limit=50)
        
        if not user_sessions:
            return suggestions
        
        # Analyze user preferences
        mode_preferences = {}
        object_preferences = {}
        
        for session in user_sessions:
            if session.processing_feedback and session.processing_feedback.preferred_mode:
                mode = session.processing_feedback.preferred_mode
                mode_preferences[mode] = mode_preferences.get(mode, 0) + 1
            
            for obj_feedback in session.object_feedbacks:
                object_type = obj_feedback.original_detection.get('label', 'unknown')
                if obj_feedback.action == FeedbackAction.CONFIRM:
                    object_preferences[object_type] = object_preferences.get(object_type, 0) + 1
        
        # Recommend mode based on preferences
        if mode_preferences:
            suggestions['recommended_mode'] = max(mode_preferences, key=mode_preferences.get)
        
        # Suggest expected objects
        if object_preferences:
            suggestions['expected_objects'] = list(object_preferences.keys())[:5]
        
        # Get confidence adjustments
        for session in user_sessions:
            for obj_feedback in session.object_feedbacks:
                if obj_feedback.confidence_rating is not None:
                    object_type = obj_feedback.original_detection.get('label', 'unknown')
                    original_confidence = obj_feedback.original_detection.get('confidence', 0.5)
                    adjustment = obj_feedback.confidence_rating - original_confidence
                    
                    if object_type not in suggestions['confidence_adjustments']:
                        suggestions['confidence_adjustments'][object_type] = []
                    
                    suggestions['confidence_adjustments'][object_type].append(adjustment)
        
        # Calculate average adjustments
        for obj_type, adjustments in suggestions['confidence_adjustments'].items():
            suggestions['confidence_adjustments'][obj_type] = np.mean(adjustments)
        
        return suggestions
    
    def _hash_image(self, image: Image.Image) -> str:
        """Generate hash for image."""
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return hashlib.md5(buffer.getvalue()).hexdigest()
    
    def _store_learning_patterns(self, patterns: Dict[str, Any]):
        """Store learned patterns in database."""
        conn = sqlite3.connect(self.database.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO learning_patterns (pattern_type, pattern_data, confidence, created_at, last_used)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                'user_feedback_insights',
                json.dumps(patterns),
                0.8,  # Confidence based on feedback quality
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing learning patterns: {e}")
        finally:
            conn.close()
    
    def get_feedback_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive feedback statistics."""
        sessions = self.database.get_feedback_history(user_id=user_id, limit=1000)
        
        if not sessions:
            return {}
        
        stats = {
            'total_sessions': len(sessions),
            'total_object_feedbacks': sum(len(s.object_feedbacks) for s in sessions),
            'feedback_types': {},
            'action_distribution': {},
            'avg_ratings': {},
            'common_issues': []
        }
        
        # Analyze feedback types and actions
        for session in sessions:
            # Processing feedback analysis
            if session.processing_feedback:
                feedback_type = session.processing_feedback.feedback_type.value
                stats['feedback_types'][feedback_type] = stats['feedback_types'].get(feedback_type, 0) + 1
                
                if session.processing_feedback.overall_rating is not None:
                    if 'overall' not in stats['avg_ratings']:
                        stats['avg_ratings']['overall'] = []
                    stats['avg_ratings']['overall'].append(session.processing_feedback.overall_rating)
            
            # Object feedback analysis
            for obj_feedback in session.object_feedbacks:
                action = obj_feedback.action.value
                stats['action_distribution'][action] = stats['action_distribution'].get(action, 0) + 1
                
                if obj_feedback.confidence_rating is not None:
                    object_type = obj_feedback.original_detection.get('label', 'unknown')
                    if object_type not in stats['avg_ratings']:
                        stats['avg_ratings'][object_type] = []
                    stats['avg_ratings'][object_type].append(obj_feedback.confidence_rating)
                
                if obj_feedback.user_comment:
                    stats['common_issues'].append(obj_feedback.user_comment)
        
        for key, ratings in stats['avg_ratings'].items():
            stats['avg_ratings'][key] = np.mean(ratings)
        
        return stats
