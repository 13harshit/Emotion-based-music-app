import pymongo
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from bson import ObjectId

class DatabaseManager:
    def __init__(self):
        self.mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.client = None
        self.db = None
        self._init_connection()
    
    def _init_connection(self):
        """Initialize MongoDB connection"""
        try:
            self.client = pymongo.MongoClient(self.mongodb_uri)
            self.db = self.client['enhanced_music_app']
            
            # Test connection
            self.client.admin.command('ismaster')
            
            # Initialize collections and indexes
            self._setup_collections()
            
        except Exception as e:
            print(f"Database connection failed: {e}")
            self.client = None
            self.db = None
    
    def _setup_collections(self):
        """Setup collections and create indexes"""
        if self.db is None:
            return
        
        # Users collection
        users = self.db['users']
        users.create_index("username", unique=True)
        users.create_index("email", unique=True)
        users.create_index("created_at")
        
        # Emotion history collection
        emotions = self.db['emotion_history']
        emotions.create_index([("username", 1), ("timestamp", -1)])
        emotions.create_index("emotion")
        emotions.create_index("timestamp")
        
        # Games history collection
        games = self.db['games_history']
        games.create_index([("username", 1), ("timestamp", -1)])
        games.create_index("game_name")
        
        # User sessions collection
        sessions = self.db['user_sessions']
        sessions.create_index([("username", 1), ("session_start", -1)])
        sessions.create_index("session_start", expireAfterSeconds=86400)  # 24 hours
        
        # Music recommendations collection
        recommendations = self.db['music_recommendations']
        recommendations.create_index([("username", 1), ("timestamp", -1)])
        recommendations.create_index("platform")
        
        # User preferences collection
        preferences = self.db['user_preferences']
        preferences.create_index("username", unique=True)
    
    def get_collection(self, collection_name: str):
        """Get a MongoDB collection"""
        if self.db is None:
            return None
        return self.db[collection_name]

# Initialize database manager
db_manager = DatabaseManager()

# User Profile Functions
def get_user_profile(username: str) -> Optional[Dict]:
    """Get user profile information"""
    try:
        users = db_manager.get_collection('users')
        if users is None:
            return None
        
        user = users.find_one({"username": username})
        if user:
            # Convert ObjectId to string for JSON serialization
            user['_id'] = str(user['_id'])
            return user
        return None
        
    except Exception as e:
        print(f"Error getting user profile: {e}")
        return None

def update_user_preferences(username: str, preferences: Dict) -> bool:
    """Update user preferences"""
    try:
        users = db_manager.get_collection('users')
        if users is None:
            return False
        
        # Update user profile with new preferences
        result = users.update_one(
            {"username": username},
            {
                "$set": {
                    f"profile.{key}": value for key, value in preferences.items()
                },
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        print(f"Error updating user preferences: {e}")
        return False

# Emotion Detection Functions
def save_emotion_detection(username: str, emotion: str, language: str = "", 
                         artist: str = "", confidence: float = 0.0) -> bool:
    """Save emotion detection result"""
    try:
        emotions = db_manager.get_collection('emotion_history')
        if emotions is None:
            return False
        
        emotion_doc = {
            "username": username,
            "emotion": emotion,
            "language": language,
            "artist": artist,
            "confidence": confidence,
            "timestamp": datetime.utcnow(),
            "session_id": f"{username}_{datetime.now().strftime('%Y%m%d')}"
        }
        
        result = emotions.insert_one(emotion_doc)
        return result.inserted_id is not None
        
    except Exception as e:
        print(f"Error saving emotion detection: {e}")
        return False

def get_emotion_history(username: str, start_date: datetime = None, 
                       end_date: datetime = None, 
                       emotion_filter: List[str] = None) -> List[Dict]:
    """Get emotion detection history for a user"""
    try:
        emotions = db_manager.get_collection('emotion_history')
        if emotions is None:
            return []
        
        # Build query
        query = {"username": username}
        
        # Add date range filter
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter["$gte"] = start_date
            if end_date:
                date_filter["$lte"] = end_date
            query["timestamp"] = date_filter
        
        # Add emotion filter
        if emotion_filter:
            query["emotion"] = {"$in": emotion_filter}
        
        # Execute query
        cursor = emotions.find(query).sort("timestamp", -1).limit(1000)
        
        history = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
            history.append(doc)
        
        return history
        
    except Exception as e:
        print(f"Error getting emotion history: {e}")
        return []

def get_emotion_statistics(username: str, days: int = 30) -> Dict:
    """Get emotion statistics for a user"""
    try:
        emotions = db_manager.get_collection('emotion_history')
        if emotions is None:
            return {}
        
        # Date range for the last N days
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        pipeline = [
            {
                "$match": {
                    "username": username,
                    "timestamp": {"$gte": start_date, "$lte": end_date}
                }
            },
            {
                "$group": {
                    "_id": "$emotion",
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence"}
                }
            },
            {
                "$sort": {"count": -1}
            }
        ]
        
        results = list(emotions.aggregate(pipeline))
        
        # Format results
        stats = {
            "emotion_counts": {doc["_id"]: doc["count"] for doc in results},
            "confidence_scores": {doc["_id"]: doc["avg_confidence"] for doc in results},
            "total_detections": sum(doc["count"] for doc in results),
            "unique_emotions": len(results),
            "most_common_emotion": results[0]["_id"] if results else None,
            "period_days": days
        }
        
        return stats
        
    except Exception as e:
        print(f"Error getting emotion statistics: {e}")
        return {}
def save_music_recommendation(username: str, platform: str, query: str, 
                            emotion: str, language: str = "", artist: str = "") -> bool:
    """Save music recommendation activity"""
    try:
        recommendations = db_manager.get_collection('music_recommendations')
        if recommendations is None:
            return False
        
        rec_doc = {
            "username": username,
            "platform": platform,
            "query": query,
            "emotion": emotion,
            "language": language,
            "artist": artist,
            "timestamp": datetime.utcnow()
        }
        
        result = recommendations.insert_one(rec_doc)
        return result.inserted_id is not None
        
    except Exception as e:
        print(f"Error saving music recommendation: {e}")
        return False
# Games Functions
def track_game_play(username: str, game_name: str, game_url: str = "", 
                   play_duration: int = 0) -> bool:
    """Track game play activity"""
    try:
        games = db_manager.get_collection('games_history')
        if games is None:
            return False
        
        game_doc = {
            "username": username,
            "game_name": game_name,
            "game_url": game_url,
            "play_duration": play_duration,  # in seconds
            "timestamp": datetime.utcnow(),
            "session_id": f"{username}_{datetime.now().strftime('%Y%m%d')}"
        }
        
        result = games.insert_one(game_doc)
        
        # Update user stats
        users = db_manager.get_collection('users')
        if users:
            users.update_one(
                {"username": username},
                {"$inc": {"stats.games_played": 1}}
            )
        
        return result.inserted_id is not None
        
    except Exception as e:
        print(f"Error tracking game play: {e}")
        return False

def get_games_history(username: str, limit: int = 100) -> List[Dict]:
    """Get games play history for a user"""
    try:
        games = db_manager.get_collection('games_history')
        if games is None:
            return []
        
        cursor = games.find({"username": username}).sort("timestamp", -1).limit(limit)
        
        history = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            history.append(doc)
        
        return history
        
    except Exception as e:
        print(f"Error getting games history: {e}")
        return []

# Music Recommendations Functions
def save_music_recommendation(username: str, platform: str, query: str, 
                            emotion: str, language: str = "", artist: str = "") -> bool:
    """Save music recommendation activity"""
    try:
        recommendations = db_manager.get_collection('music_recommendations')
        if recommendations is None:
            return False
        
        rec_doc = {
            "username": username,
            "platform": platform,
            "query": query,
            "emotion": emotion,
            "language": language,
            "artist": artist,
            "timestamp": datetime.utcnow()
        }
        
        result = recommendations.insert_one(rec_doc)
        return result.inserted_id is not None
        
    except Exception as e:
        print(f"Error saving music recommendation: {e}")
        return False

