"""
Semantic Search Engine for Vachanamrut Discourses
Advanced semantic search using sentence transformers and FAISS indexing

This engine provides:
1. Semantic embeddings for all discourse content
2. Fast similarity search using FAISS
3. Context-aware ranking for LLM responses
4. Spiritual theme-based filtering
5. Multi-modal search (text + metadata)
"""

import numpy as np
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import hashlib
import time
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("WARNING: sentence-transformers or faiss-cpu not installed. Falling back to basic similarity.")

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from semantic search"""
    discourse_id: str
    reference: str
    content: str
    title: str
    chapter: str
    similarity_score: float
    semantic_score: float
    metadata_score: float
    combined_score: float
    page_number: int
    word_count: int
    spiritual_themes: List[str]
    explanation: str  # Why this result was chosen


@dataclass
class SearchQuery:
    """Structured search query with context"""
    query_text: str
    theme_filter: Optional[str] = None
    chapter_filter: Optional[str] = None
    max_results: int = 10
    min_similarity: float = 0.3
    boost_keywords: List[str] = None
    search_mode: str = "semantic"  # "semantic", "keyword", "hybrid"


class VachanamrutSemanticSearch:
    """
    Advanced semantic search engine for Vachanamrut discourses
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.discourses = []
        self.embeddings = None
        self.discourse_lookup = {}  # Fast ID to discourse mapping
        
        # Search configuration
        self.config = {
            'embedding_dim': 384,  # all-MiniLM-L6-v2 dimensions
            'index_type': 'flat',  # 'flat' or 'ivf' for large datasets
            'similarity_threshold': 0.3,
            'max_results_default': 10
        }
        
        # Spiritual context keywords for boosting
        self.spiritual_keywords = {
            'high_priority': [
                'god', 'bhagwan', 'swaminarayan', 'divine', 'supreme',
                'liberation', 'moksha', 'salvation', 'enlightenment'
            ],
            'medium_priority': [
                'dharma', 'righteousness', 'devotion', 'faith', 'worship',
                'meditation', 'prayer', 'spiritual', 'soul', 'atma'
            ],
            'context_keywords': [
                'assembly', 'discourse', 'teaching', 'guidance', 'wisdom',
                'devotee', 'sant', 'sadhu', 'disciple', 'follower'
            ]
        }
        
        self.is_initialized = False

    def initialize(self, discourse_data: List[Dict]) -> bool:
        """
        Initialize the semantic search engine with discourse data
        """
        try:
            if not DEPENDENCIES_AVAILABLE:
                logger.warning("Dependencies not available. Using fallback search.")
                return self._initialize_fallback(discourse_data)
            
            logger.info("Initializing semantic search engine...")
            
            # Load sentence transformer model
            self.model = SentenceTransformer(f'sentence-transformers/{self.model_name}')
            logger.info(f"Loaded model: {self.model_name}")
            
            # Prepare discourse data
            self.discourses = discourse_data
            self._build_discourse_lookup()
            
            # Generate embeddings
            self._generate_embeddings()
            
            # Build FAISS index
            self._build_search_index()
            
            self.is_initialized = True
            logger.info(f"Semantic search initialized with {len(self.discourses)} discourses")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic search: {e}")
            return self._initialize_fallback(discourse_data)

    def _initialize_fallback(self, discourse_data: List[Dict]) -> bool:
        """
        Initialize fallback search without embeddings
        """
        self.discourses = discourse_data
        self._build_discourse_lookup()
        self.is_initialized = True
        logger.info(f"Fallback search initialized with {len(self.discourses)} discourses")
        return True

    def _build_discourse_lookup(self):
        """
        Build fast lookup table for discourses
        """
        self.discourse_lookup = {}
        for i, discourse in enumerate(self.discourses):
            self.discourse_lookup[discourse.get('id', f'discourse_{i}')] = {
                'index': i,
                'discourse': discourse
            }

    def _generate_embeddings(self):
        """
        Generate embeddings for all discourse content
        """
        logger.info("Generating semantic embeddings...")
        
        # Prepare text for embedding
        texts_to_embed = []
        for discourse in self.discourses:
            # Combine content with title and themes for richer embeddings
            content = discourse.get('content', '')
            title = discourse.get('title', '')
            themes = ' '.join(discourse.get('spiritual_themes', []))
            
            # Create enriched text for embedding
            enriched_text = f"{title} {content} {themes}".strip()
            texts_to_embed.append(enriched_text)
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=True)
            embeddings.extend(batch_embeddings)
        
        self.embeddings = np.array(embeddings, dtype=np.float32)
        logger.info(f"Generated {len(embeddings)} embeddings of dimension {self.embeddings.shape[1]}")

    def _build_search_index(self):
        """
        Build FAISS index for fast similarity search
        """
        logger.info("Building FAISS search index...")
        
        embedding_dim = self.embeddings.shape[1]
        
        if len(self.discourses) < 1000:
            # Use flat index for smaller datasets (exact search)
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity
        else:
            # Use IVF index for larger datasets (approximate search)
            quantizer = faiss.IndexFlatIP(embedding_dim)
            nlist = min(100, int(np.sqrt(len(self.discourses))))  # Number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.embeddings)
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        self.index.add(normalized_embeddings)
        logger.info(f"Built search index with {self.index.ntotal} vectors")

    def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform semantic search for the given query
        """
        if not self.is_initialized:
            raise RuntimeError("Search engine not initialized")
        
        if not DEPENDENCIES_AVAILABLE or self.model is None:
            return self._fallback_search(query)
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query.query_text])
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Perform similarity search
            similarities, indices = self.index.search(
                query_embedding.astype(np.float32), 
                min(query.max_results * 2, len(self.discourses))  # Get more candidates for filtering
            )
            
            # Process results
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx >= len(self.discourses) or similarity < query.min_similarity:
                    continue
                
                discourse = self.discourses[idx]
                
                # Apply filters
                if not self._passes_filters(discourse, query):
                    continue
                
                # Calculate enhanced scores
                result = self._create_search_result(discourse, similarity, query)
                results.append(result)
            
            # Sort by combined score and limit results
            results.sort(key=lambda x: x.combined_score, reverse=True)
            return results[:query.max_results]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return self._fallback_search(query)

    def _passes_filters(self, discourse: Dict, query: SearchQuery) -> bool:
        """
        Check if discourse passes query filters
        """
        # Theme filter
        if query.theme_filter:
            discourse_themes = [t.lower() for t in discourse.get('spiritual_themes', [])]
            if query.theme_filter.lower() not in discourse_themes:
                return False
        
        # Chapter filter
        if query.chapter_filter:
            discourse_chapter = discourse.get('chapter', '').lower()
            if query.chapter_filter.lower() not in discourse_chapter:
                return False
        
        return True

    def _create_search_result(self, discourse: Dict, similarity: float, query: SearchQuery) -> SearchResult:
        """
        Create enhanced search result with multiple scoring factors
        """
        # Base similarity score
        semantic_score = float(similarity)
        
        # Metadata score based on content quality and completeness
        metadata_score = self._calculate_metadata_score(discourse, query)
        
        # Keyword boost score
        keyword_score = self._calculate_keyword_score(discourse, query)
        
        # Combined score with weights
        combined_score = (
            semantic_score * 0.6 +
            metadata_score * 0.25 +
            keyword_score * 0.15
        )
        
        # Generate explanation
        explanation = self._generate_explanation(discourse, semantic_score, metadata_score, keyword_score)
        
        return SearchResult(
            discourse_id=discourse.get('id', ''),
            reference=discourse.get('reference', ''),
            content=discourse.get('content', ''),
            title=discourse.get('title', ''),
            chapter=discourse.get('chapter', ''),
            similarity_score=semantic_score,
            semantic_score=semantic_score,
            metadata_score=metadata_score,
            combined_score=combined_score,
            page_number=discourse.get('page_number', 0),
            word_count=discourse.get('word_count', 0),
            spiritual_themes=discourse.get('spiritual_themes', []),
            explanation=explanation
        )

    def _calculate_metadata_score(self, discourse: Dict, query: SearchQuery) -> float:
        """
        Calculate score based on discourse metadata quality
        """
        score = 0.0
        
        # Content quality indicators
        quality_score = discourse.get('quality_score', 0.5)
        score += quality_score * 0.4
        
        # Content length (longer content often more comprehensive)
        word_count = discourse.get('word_count', 0)
        if word_count > 500:
            score += 0.3
        elif word_count > 200:
            score += 0.2
        else:
            score += 0.1
        
        # Has title
        if discourse.get('title'):
            score += 0.15
        
        # Has date information
        if discourse.get('date_info'):
            score += 0.15
        
        return min(score, 1.0)

    def _calculate_keyword_score(self, discourse: Dict, query: SearchQuery) -> float:
        """
        Calculate score based on keyword matches and boosts
        """
        score = 0.0
        content = discourse.get('content', '').lower()
        title = discourse.get('title', '').lower()
        themes = ' '.join(discourse.get('spiritual_themes', [])).lower()
        
        # Query keyword matches
        query_words = query.query_text.lower().split()
        for word in query_words:
            if word in content:
                score += 0.1
            if word in title:
                score += 0.2
            if word in themes:
                score += 0.15
        
        # Spiritual keyword boosts
        for keyword in self.spiritual_keywords['high_priority']:
            if keyword in content or keyword in title:
                score += 0.1
        
        for keyword in self.spiritual_keywords['medium_priority']:
            if keyword in content:
                score += 0.05
        
        # Boost keywords from query
        if query.boost_keywords:
            for boost_word in query.boost_keywords:
                if boost_word.lower() in content:
                    score += 0.2
        
        return min(score, 1.0)

    def _generate_explanation(self, discourse: Dict, semantic_score: float, 
                            metadata_score: float, keyword_score: float) -> str:
        """
        Generate human-readable explanation for why this result was chosen
        """
        explanations = []
        
        if semantic_score > 0.7:
            explanations.append("high semantic similarity")
        elif semantic_score > 0.5:
            explanations.append("good semantic match")
        
        if metadata_score > 0.7:
            explanations.append("high-quality content")
        
        if keyword_score > 0.5:
            explanations.append("strong keyword relevance")
        
        if discourse.get('spiritual_themes'):
            explanations.append(f"covers themes: {', '.join(discourse['spiritual_themes'][:3])}")
        
        return "; ".join(explanations) if explanations else "general relevance"

    def _fallback_search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Fallback search using basic text matching
        """
        logger.info("Using fallback text-based search")
        
        results = []
        query_words = set(query.query_text.lower().split())
        
        for discourse in self.discourses:
            if not self._passes_filters(discourse, query):
                continue
            
            content = discourse.get('content', '').lower()
            title = discourse.get('title', '').lower()
            
            # Simple keyword matching
            content_words = set(content.split())
            title_words = set(title.split())
            
            # Calculate overlap
            content_overlap = len(query_words.intersection(content_words))
            title_overlap = len(query_words.intersection(title_words))
            
            if content_overlap == 0 and title_overlap == 0:
                continue
            
            # Simple scoring
            similarity = (content_overlap + title_overlap * 2) / (len(query_words) * 3)
            
            if similarity >= query.min_similarity:
                result = SearchResult(
                    discourse_id=discourse.get('id', ''),
                    reference=discourse.get('reference', ''),
                    content=discourse.get('content', ''),
                    title=discourse.get('title', ''),
                    chapter=discourse.get('chapter', ''),
                    similarity_score=similarity,
                    semantic_score=similarity,
                    metadata_score=0.5,
                    combined_score=similarity,
                    page_number=discourse.get('page_number', 0),
                    word_count=discourse.get('word_count', 0),
                    spiritual_themes=discourse.get('spiritual_themes', []),
                    explanation=f"keyword matches: {content_overlap + title_overlap}"
                )
                results.append(result)
        
        # Sort and limit
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:query.max_results]

    def get_contextual_passages(self, query: str, theme: str = None, 
                              max_results: int = 5) -> List[Dict]:
        """
        Get contextual passages for LLM integration
        """
        search_query = SearchQuery(
            query_text=query,
            theme_filter=theme,
            max_results=max_results,
            min_similarity=0.3,
            search_mode="semantic"
        )
        
        results = self.search(search_query)
        
        # Format for LLM context
        contextual_passages = []
        for result in results:
            passage = {
                'reference': result.reference,
                'content': result.content,
                'relevance_score': result.combined_score,
                'explanation': result.explanation,
                'themes': result.spiritual_themes,
                'chapter': result.chapter
            }
            contextual_passages.append(passage)
        
        return contextual_passages

    def save_index(self, save_path: str) -> bool:
        """
        Save the search index and embeddings for later use
        """
        try:
            save_dir = Path(save_path)
            save_dir.mkdir(exist_ok=True)
            
            # Save FAISS index
            if self.index and DEPENDENCIES_AVAILABLE:
                faiss.write_index(self.index, str(save_dir / "search_index.faiss"))
            
            # Save embeddings and metadata
            save_data = {
                'embeddings': self.embeddings.tolist() if self.embeddings is not None else None,
                'discourses': self.discourses,
                'config': self.config,
                'model_name': self.model_name,
                'created_at': datetime.now().isoformat()
            }
            
            with open(save_dir / "search_data.json", 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Search index saved to {save_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save search index: {e}")
            return False

    def load_index(self, load_path: str) -> bool:
        """
        Load previously saved search index
        """
        try:
            load_dir = Path(load_path)
            
            # Load metadata and embeddings
            with open(load_dir / "search_data.json", 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            
            self.discourses = save_data['discourses']
            self.config = save_data['config']
            self.model_name = save_data['model_name']
            
            if save_data['embeddings']:
                self.embeddings = np.array(save_data['embeddings'], dtype=np.float32)
            
            # Load FAISS index
            if DEPENDENCIES_AVAILABLE and (load_dir / "search_index.faiss").exists():
                self.index = faiss.read_index(str(load_dir / "search_index.faiss"))
                
                # Load model if needed
                if not self.model:
                    self.model = SentenceTransformer(f'sentence-transformers/{self.model_name}')
            
            self._build_discourse_lookup()
            self.is_initialized = True
            
            logger.info(f"Search index loaded from {load_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load search index: {e}")
            return False

    def get_search_stats(self) -> Dict:
        """
        Get search engine statistics
        """
        stats = {
            'initialized': self.is_initialized,
            'total_discourses': len(self.discourses),
            'model_name': self.model_name,
            'has_embeddings': self.embeddings is not None,
            'has_index': self.index is not None,
            'dependencies_available': DEPENDENCIES_AVAILABLE
        }
        
        if self.embeddings is not None:
            stats['embedding_shape'] = self.embeddings.shape
        
        if self.index is not None:
            stats['index_size'] = self.index.ntotal
        
        return stats


def create_semantic_search_engine(discourse_data_path: str) -> VachanamrutSemanticSearch:
    """
    Create and initialize semantic search engine from discourse data
    """
    # Load discourse data
    with open(discourse_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    discourses = data.get('passages', [])
    
    # Create and initialize search engine
    search_engine = VachanamrutSemanticSearch()
    
    if search_engine.initialize(discourses):
        logger.info("Semantic search engine created successfully")
        return search_engine
    else:
        raise RuntimeError("Failed to create semantic search engine")


def test_semantic_search():
    """
    Test the semantic search engine
    """
    print("=== TESTING SEMANTIC SEARCH ENGINE ===")
    
    # Load the working processor data
    data_path = "/Users/darshen/Documents/vachanamrut-companion/backend/data/processed/working_vachanamrut.json"
    
    try:
        search_engine = create_semantic_search_engine(data_path)
        
        # Test searches
        test_queries = [
            "How can I develop devotion to God?",
            "What is the nature of the soul?",
            "How should one meditate?",
            "What are the qualities of a true devotee?",
            "How to overcome desires and attachments?"
        ]
        
        print(f"Search engine stats: {search_engine.get_search_stats()}")
        print()
        
        for query_text in test_queries:
            print(f"Query: '{query_text}'")
            query = SearchQuery(
                query_text=query_text,
                max_results=3,
                min_similarity=0.2
            )
            
            results = search_engine.search(query)
            
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.reference} (score: {result.combined_score:.2f})")
                print(f"   {result.explanation}")
                print(f"   Content: {result.content[:100]}...")
                print()
        
        return True
        
    except Exception as e:
        print(f"Semantic search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_semantic_search()