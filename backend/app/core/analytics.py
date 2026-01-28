"""
Analytics Service for Vachanamrut Study Companion
Tracks usage metrics and collects user feedback
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import Counter
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class ChatInteraction:
    """Represents a single chat interaction"""
    id: str
    timestamp: str
    query: str
    query_length: int
    response_length: int
    theme: Optional[str]
    citations_count: int
    related_themes: List[str]
    response_time_ms: int
    search_method: str  # semantic, text, fallback
    feedback_score: Optional[int] = None  # 1-5 rating
    feedback_text: Optional[str] = None


@dataclass
class UsageMetrics:
    """Aggregated usage metrics"""
    total_queries: int = 0
    total_sessions: int = 0
    avg_query_length: float = 0.0
    avg_response_time_ms: float = 0.0
    theme_distribution: Dict[str, int] = field(default_factory=dict)
    popular_queries: List[str] = field(default_factory=list)
    feedback_avg: float = 0.0
    feedback_count: int = 0
    search_method_distribution: Dict[str, int] = field(default_factory=dict)
    hourly_distribution: Dict[int, int] = field(default_factory=dict)


class AnalyticsService:
    """
    Service for tracking usage analytics and collecting feedback.
    Stores data locally for MVP; can be extended to external services.
    """
    
    def __init__(self, storage_path: str = "data/analytics"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory buffers for performance
        self.interactions: List[ChatInteraction] = []
        self.metrics = UsageMetrics()
        
        # Buffer settings
        self.buffer_size = 100  # Flush to disk every 100 interactions
        self.flush_interval = 300  # Flush every 5 minutes
        self.last_flush = time.time()
        
        # Thread lock for concurrent access
        self._lock = threading.Lock()
        
        # Load existing data
        self._load_existing_data()
    
    def track_interaction(
        self,
        query: str,
        response: str,
        theme: Optional[str],
        citations_count: int,
        related_themes: List[str],
        response_time_ms: int,
        search_method: str = "semantic"
    ) -> str:
        """Track a chat interaction"""
        interaction_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{len(self.interactions)}"
        
        interaction = ChatInteraction(
            id=interaction_id,
            timestamp=datetime.utcnow().isoformat(),
            query=query[:500],  # Limit stored query length
            query_length=len(query),
            response_length=len(response),
            theme=theme,
            citations_count=citations_count,
            related_themes=related_themes[:5],
            response_time_ms=response_time_ms,
            search_method=search_method
        )
        
        with self._lock:
            self.interactions.append(interaction)
            self._update_metrics(interaction)
            
            # Check if we need to flush
            if len(self.interactions) >= self.buffer_size or \
               time.time() - self.last_flush > self.flush_interval:
                self._flush_to_disk()
        
        logger.debug(f"Tracked interaction: {interaction_id}")
        return interaction_id
    
    def record_feedback(
        self,
        interaction_id: str,
        score: int,
        feedback_text: Optional[str] = None
    ) -> bool:
        """Record user feedback for an interaction"""
        if not 1 <= score <= 5:
            logger.warning(f"Invalid feedback score: {score}")
            return False
        
        with self._lock:
            # Find the interaction
            for interaction in self.interactions:
                if interaction.id == interaction_id:
                    interaction.feedback_score = score
                    interaction.feedback_text = feedback_text
                    
                    # Update feedback metrics
                    self.metrics.feedback_count += 1
                    total_feedback = self.metrics.feedback_avg * (self.metrics.feedback_count - 1)
                    self.metrics.feedback_avg = (total_feedback + score) / self.metrics.feedback_count
                    
                    logger.info(f"Recorded feedback for {interaction_id}: {score}/5")
                    return True
        
        # Check persisted data
        return self._update_persisted_feedback(interaction_id, score, feedback_text)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current usage metrics"""
        with self._lock:
            return {
                "total_queries": self.metrics.total_queries,
                "total_sessions": self.metrics.total_sessions,
                "avg_query_length": round(self.metrics.avg_query_length, 1),
                "avg_response_time_ms": round(self.metrics.avg_response_time_ms, 1),
                "theme_distribution": dict(self.metrics.theme_distribution),
                "popular_themes": self._get_top_items(self.metrics.theme_distribution, 5),
                "feedback_avg": round(self.metrics.feedback_avg, 2),
                "feedback_count": self.metrics.feedback_count,
                "search_methods": dict(self.metrics.search_method_distribution),
                "peak_hours": self._get_peak_hours(),
                "recent_queries": len(self.interactions)
            }
    
    def get_daily_summary(self) -> Dict[str, Any]:
        """Get summary for the current day"""
        today = datetime.utcnow().date().isoformat()
        today_file = self.storage_path / f"interactions_{today}.json"
        
        summary = {
            "date": today,
            "total_queries": 0,
            "avg_response_time": 0,
            "top_themes": [],
            "feedback_summary": {"positive": 0, "neutral": 0, "negative": 0}
        }
        
        interactions = self.interactions.copy()
        
        # Also load from today's file
        if today_file.exists():
            try:
                with open(today_file, 'r') as f:
                    stored = json.load(f)
                    interactions.extend([
                        ChatInteraction(**i) for i in stored.get('interactions', [])
                    ])
            except Exception as e:
                logger.warning(f"Could not load today's data: {e}")
        
        if not interactions:
            return summary
        
        # Calculate summary
        summary["total_queries"] = len(interactions)
        
        total_time = sum(i.response_time_ms for i in interactions)
        summary["avg_response_time"] = round(total_time / len(interactions), 1)
        
        theme_counts = Counter(i.theme for i in interactions if i.theme)
        summary["top_themes"] = [t for t, _ in theme_counts.most_common(5)]
        
        for i in interactions:
            if i.feedback_score:
                if i.feedback_score >= 4:
                    summary["feedback_summary"]["positive"] += 1
                elif i.feedback_score <= 2:
                    summary["feedback_summary"]["negative"] += 1
                else:
                    summary["feedback_summary"]["neutral"] += 1
        
        return summary
    
    def export_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        """Export analytics data for a date range"""
        # Default to last 7 days
        if not end_date:
            end_date = datetime.utcnow().date().isoformat()
        if not start_date:
            start = datetime.utcnow() - timedelta(days=7)
            start_date = start.date().isoformat()
        
        exported = {
            "date_range": {"start": start_date, "end": end_date},
            "metrics": self.get_metrics(),
            "interactions": []
        }
        
        # Collect interactions from files
        current = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        while current <= end:
            date_str = current.date().isoformat()
            file_path = self.storage_path / f"interactions_{date_str}.json"
            
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        exported["interactions"].extend(data.get('interactions', []))
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
            
            current += timedelta(days=1)
        
        # Add in-memory interactions
        for interaction in self.interactions:
            exported["interactions"].append(asdict(interaction))
        
        return exported
    
    def _update_metrics(self, interaction: ChatInteraction):
        """Update running metrics with new interaction"""
        self.metrics.total_queries += 1
        
        # Update averages
        n = self.metrics.total_queries
        self.metrics.avg_query_length = (
            (self.metrics.avg_query_length * (n - 1) + interaction.query_length) / n
        )
        self.metrics.avg_response_time_ms = (
            (self.metrics.avg_response_time_ms * (n - 1) + interaction.response_time_ms) / n
        )
        
        # Update theme distribution
        if interaction.theme:
            self.metrics.theme_distribution[interaction.theme] = \
                self.metrics.theme_distribution.get(interaction.theme, 0) + 1
        
        # Update search method distribution
        self.metrics.search_method_distribution[interaction.search_method] = \
            self.metrics.search_method_distribution.get(interaction.search_method, 0) + 1
        
        # Update hourly distribution
        hour = datetime.fromisoformat(interaction.timestamp).hour
        self.metrics.hourly_distribution[hour] = \
            self.metrics.hourly_distribution.get(hour, 0) + 1
    
    def _flush_to_disk(self):
        """Persist buffered data to disk"""
        if not self.interactions:
            return
        
        try:
            today = datetime.utcnow().date().isoformat()
            file_path = self.storage_path / f"interactions_{today}.json"
            
            # Load existing data
            existing_data = {"interactions": [], "metrics": {}}
            if file_path.exists():
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
            
            # Append new interactions
            for interaction in self.interactions:
                existing_data["interactions"].append(asdict(interaction))
            
            # Update metrics
            existing_data["metrics"] = asdict(self.metrics)
            existing_data["last_updated"] = datetime.utcnow().isoformat()
            
            # Write back
            with open(file_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            # Clear buffer
            self.interactions = []
            self.last_flush = time.time()
            
            logger.info(f"Flushed analytics to {file_path}")
            
        except Exception as e:
            logger.error(f"Error flushing analytics: {e}")
    
    def _load_existing_data(self):
        """Load metrics from existing data files"""
        try:
            metrics_file = self.storage_path / "metrics_summary.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics.total_queries = data.get('total_queries', 0)
                    self.metrics.total_sessions = data.get('total_sessions', 0)
                    self.metrics.feedback_avg = data.get('feedback_avg', 0.0)
                    self.metrics.feedback_count = data.get('feedback_count', 0)
                    logger.info(f"Loaded existing metrics: {self.metrics.total_queries} queries")
        except Exception as e:
            logger.warning(f"Could not load existing metrics: {e}")
    
    def _update_persisted_feedback(self, interaction_id: str, score: int, text: Optional[str]) -> bool:
        """Update feedback in persisted data"""
        # Parse date from interaction ID
        try:
            date_str = f"{interaction_id[:4]}-{interaction_id[4:6]}-{interaction_id[6:8]}"
            file_path = self.storage_path / f"interactions_{date_str}.json"
            
            if not file_path.exists():
                return False
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for interaction in data.get('interactions', []):
                if interaction.get('id') == interaction_id:
                    interaction['feedback_score'] = score
                    interaction['feedback_text'] = text
                    
                    with open(file_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating persisted feedback: {e}")
            return False
    
    def _get_top_items(self, distribution: Dict[str, int], n: int) -> List[str]:
        """Get top n items from a distribution"""
        sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_items[:n]]
    
    def _get_peak_hours(self) -> List[int]:
        """Get peak usage hours"""
        if not self.metrics.hourly_distribution:
            return []
        sorted_hours = sorted(
            self.metrics.hourly_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [hour for hour, _ in sorted_hours[:3]]
    
    def shutdown(self):
        """Flush data and cleanup on shutdown"""
        with self._lock:
            self._flush_to_disk()
            
            # Save metrics summary
            metrics_file = self.storage_path / "metrics_summary.json"
            with open(metrics_file, 'w') as f:
                json.dump(asdict(self.metrics), f, indent=2)
            
            logger.info("Analytics service shutdown complete")


# Global instance
_analytics_instance: Optional[AnalyticsService] = None


def get_analytics_service() -> AnalyticsService:
    """Get global analytics service instance"""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = AnalyticsService()
    return _analytics_instance

