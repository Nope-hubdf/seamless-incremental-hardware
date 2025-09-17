#!/usr/bin/env python3
"""
HOLIBOT DEFINITIEVE FIX - Gebaseerd op Historische Analyse
Datum: 12 September 2025
Versie: FINAL voor investeerders

Dit script combineert ALLE werkende elementen uit augustus 2025
en verwijdert ALLE problematische toevoegingen uit september.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="HoliBot Tourism API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
data = None
model = None
embeddings_matrix = None

# Request models - GEEN SearchResult class!
class QueryRequest(BaseModel):
    question: str  # LET OP: "question" niet "query"
    max_results: int = 5
    min_score: float = 0.45  # Verlaagde threshold

class QueryResponse(BaseModel):
    query: str
    results: List[Dict]  # Gewone dictionaries
    response_time_ms: float
    total_results: int
    timestamp: str

def load_data():
    """Load vector database and model - BEWEZEN WERKEND"""
    global data, model, embeddings_matrix
    
    try:
        logger.info("Loading vector database...")
        with open('data/holibot_week2_local_20250813_152511.pkl', 'rb') as f:
            data = pickle.load(f)
        
        logger.info("Loading SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("Creating embeddings matrix...")
        embeddings_matrix = np.array(data['vectors'])
        
        logger.info(f"SUCCESS: Loaded {len(data['vectors'])} vectors")
        return True
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return False

def semantic_search(query: str, max_results: int = 5, min_score: float = 0.45) -> List[Dict]:
    """
    DEFINITIEVE WERKENDE VERSIE - Gebaseerd op 13 augustus implementatie
    """
    if data is None or model is None:
        return []
    
    try:
        # CRUCIALE FIX: query in array voor model.encode
        query_embedding = model.encode([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
        
        # Get indices of scores above threshold
        valid_indices = np.where(similarities >= min_score)[0]
        
        # Create results - GEEN SearchResult class
        results = []
        for idx in valid_indices:
            score = float(similarities[idx])
            metadata = data['metadata'][idx]
            
            # Return dictionary format
            results.append({
                'score': score,
                'poi_name': metadata.get('poi_name', 'Unknown'),
                'answer': metadata.get('answer', ''),
                'question': metadata.get('question', ''),
                'category': metadata.get('category', 'general')
            })
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top N results
        return results[:max_results]
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return []

@app.on_event("startup")
async def startup():
    """Load data on startup"""
    if not load_data():
        logger.error("Failed to load data on startup")

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "HoliBot Tourism API",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": ["/health", "/search", "/docs"]
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "loaded" if data is not None else "not loaded",
        "model": "loaded" if model is not None else "not loaded",
        "vectors": len(data['vectors']) if data else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/search", response_model=QueryResponse)
def search(request: QueryRequest):
    """
    HOOFDENDPOINT - Semantic search
    Parameter: "question" (niet "query"!)
    """
    start_time = time.time()
    
    # Perform semantic search
    search_results = semantic_search(
        request.question, 
        request.max_results, 
        request.min_score
    )
    
    # Calculate response time
    response_time = (time.time() - start_time) * 1000
    
    # Return response
    return QueryResponse(
        query=request.question,
        results=search_results,  # Lijst van dictionaries
        response_time_ms=round(response_time, 2),
        total_results=len(search_results),
        timestamp=datetime.now().isoformat()
    )

@app.post("/search/autonomous")
def search_autonomous(request: QueryRequest):
    """
    Fallback endpoint - zelfde als /search
    """
    return search(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)