import string

import faiss
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from typing import List

nltk.download("punkt")
nltk.download("stopwords")


class FAISSManager:
    """
    Manages FAISS index creation and searching.
    """
    def __init__(self, corpus_embeddings):
        """
        Initializes the FAISSManager with given embeddings.

        Args:
            corpus_embeddings (np.ndarray): Embeddings to index with FAISS.
        """
        self.corpus_embeddings = corpus_embeddings
        self.index = self._build_index()

    def _build_index(self)->faiss.IndexFlatL2:
        """
        Builds the FAISS index.

        Returns:
            faiss.IndexFlatL2: Initialized FAISS index.
        """
        d = self.corpus_embeddings.shape[1]  # dimension of embeddings
        index = faiss.IndexFlatL2(d)
        index.add(np.array(self.corpus_embeddings))
        return index
    
    def scores(self, query_embedding, indices):
        """
        Calculates similarity scores using FAISS.

        Args:
            query_embedding (np.ndarray): Embedding of the query.
            indices (np.ndarray): Indices of top N results.

        Returns:
            list: List of similarity scores.
        """
        faiss_scores = []
        for i in indices:
            corpus_embedding = np.expand_dims(self.corpus_embeddings[i], axis=0)
            score = util.pytorch_cos_sim(
                query_embedding, 
                corpus_embedding
            )
            score = score.numpy().flatten()[0]
            faiss_scores.append(score)
        return faiss_scores
    
    def search(self, query_embedding, top_n: int = 10):
        """
        Searches for the top N similar embeddings.

        Args:
            query_embedding (np.ndarray): Embedding of the query.
            top_n (int): Number of top results to return.

        Returns:
            tuple: Indices and similarity scores of top N results.
        """
        _, indices = self.index.search(query_embedding, top_n)
        indices = indices.flatten()
        scores = self.scores(query_embedding, indices)
        
        return indices, scores


class HybridSearch:
    """
    HybridSearch class for performing hybrid search combining BM25 and Sentence Transformers with FAISS.
    """
    
    def __init__(
        self,
        base_mapping: dict,
        transformer_model: str = "uonyeka/bge-base-financial-matryoshka",
    ):
        """
        Initializes the HybridSearch class.

        Args:
            base_mapping (dict): Dictionary containing the base mapping.
            transformer_model (str): Name of the transformer model to use.
        """
        
        self.base_mapping = base_mapping
        self.corpus = list(base_mapping.keys())
        
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        self.model = SentenceTransformer(transformer_model)
        self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=False)
        self.fmgr = FAISSManager(self.corpus_embeddings)

    def _tokenize(self, text: str) -> list:
        """
        Tokenizes and preprocesses text.

        Args:
            text (str): Text to preprocess.

        Returns:
            list: Tokenized and preprocessed text.
        """
        text = text.lower()  # Convert to lowercase
        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )  # Remove punctuation
        tokens = word_tokenize(text)  # Tokenize
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]  # Stem words
        return tokens
    
    def _normalize_scores(self, rscores)->np.ndarray:
        """
        Normalizes scores.

        Args:
            rscores (list): List of raw scores.

        Returns:
            np.ndarray: Normalized scores.
        """
        mxscr = np.max(rscores)
        mnscr = np.min(rscores)
        
        if mxscr != mnscr:
            norm_scr = (rscores - mnscr) / (mxscr - mnscr)
        else:
            norm_scr = np.array(rscores)
            
        return norm_scr
        
    def _hybrid_scores(
        self, 
        bm25_scores, 
        tfr_scores,
        transformer_weight: float = 0.9,
        bm25_weight: float = 0.3,   
    )->np.ndarray:
        """
        Combines BM25 and transformer scores with weighting.

        Args:
            bm25_scores (list): BM25 scores.
            tfr_scores (list): Transformer scores.
            transformer_weight (float): Weight for transformer scores.
            bm25_weight (float): Weight for BM25 scores.

        Returns:
            np.ndarray: Combined scores.
        """
        
         # Normalize BM25 scores and transformer scores
        bm25_scores = self._normalize_scores(bm25_scores)
        tfr_scores = self._normalize_scores(tfr_scores)

        agreement_factor = np.multiply(bm25_scores, tfr_scores) 
        bm25_weighted_scores = bm25_weight * bm25_scores
        tfr_weighted_scores = transformer_weight * tfr_scores
        
        combined_scores = ( 
            bm25_weighted_scores + 
            tfr_weighted_scores + 
            agreement_factor
        )
        
        return combined_scores
    
    def _query_embedding(self, query: str)->np.ndarray:
        """
        Encodes a query to get its embedding.

        Args:
            query (str): Query string.

        Returns:
            np.ndarray: Query embedding.
        """
        embedding = self.model.encode(query, convert_to_tensor=False)
        embedding = embedding.reshape(1, -1)
        return embedding
 
    def _ranked_result(self, candidates: List[str], scores: List[float])->List[dict]:
        """
        Creates ranked results with scores.

        Args:
            candidates (list): List of candidate strings.
            scores (list): List of scores corresponding to candidates.

        Returns:
            list: Ranked results.
        """
        # Create results with scores using a normal loop
        ranked_indices = np.argsort(scores)[::-1]
        ranked_results = []
        for i in ranked_indices:
            result = {
                "similar_name": candidates[i],
                "account_name": self.base_mapping[candidates[i]],
                "scores": scores[i],
            }
            ranked_results.append(result)
        return ranked_results
    
    
    def bm25_search(self, query: str, top_n: int = 10)->List[dict]:
        """
        Performs BM25 search.

        Args:
            query (str): Query string.
            top_n (int): Number of top results to return.

        Returns:
            list: Ranked results from BM25 search.
        """
        # Preprocess the query
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get the top N candidates from BM25
        indices = np.argsort(scores)[::-1][:top_n]
        candidates = [self.corpus[i] for i in indices]
        scores = [scores[i] for i in indices]

        # Create results with scores
        ranked_results = self._ranked_result(candidates, scores)
        return ranked_results
    
    def transformer_search(self, query: str, top_n: int = 10)->List[dict]:
        """
        Performs transformer-based search using FAISS.

        Args:
            query (str): Query string.
            top_n (int): Number of top results to return.

        Returns:
            list: Ranked results from transformer search.
        """
        query_embedding = self._query_embedding(query)
        indices, scores = self.fmgr.search(query_embedding, top_n=top_n)
        
        # Get the top N candidates
        candidates = [self.corpus[i] for i in indices]

        # Create results with scores
        ranked_results = self._ranked_result(candidates, scores)

        return ranked_results

    def hybrid_search(
        self,
        query: str,
        top_n: int = 10,
        transformer_weight: float = 0.9,
        bm25_weight: float = 0.3,
    )->List[dict]:
        """
        Performs hybrid search combining BM25 and transformer-based search.

        Args:
            query (str): Query string.
            top_n (int): Number of top results to return.
            transformer_weight (float): Weight for transformer scores.
            bm25_weight (float): Weight for BM25 scores.

        Returns:
            list: Ranked results from hybrid search.
        """
        query_embedding = self._query_embedding(query)
        tfr_indices, tfr_scores = self.fmgr.search(query_embedding, top_n=top_n)
        tfr_candidates = [self.corpus[i] for i in tfr_indices]

        # Preprocess the query for BM25
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get BM25 scores for the top N candidates
        bm25_scores = [
            bm25_scores[self.corpus.index(candidate)] 
            for candidate in tfr_candidates
        ]

        combined_scores = self._hybrid_scores(
            bm25_scores,
            tfr_scores,
            transformer_weight=transformer_weight,
            bm25_weight=bm25_weight
        )
        ranked_results = self._ranked_result(tfr_candidates, combined_scores)

        return ranked_results


# Function to evaluate search accuracy and return detailed results as a DataFrame
def evaluate_search_accuracy(test_mapping_dict, search_engine, search_method):
    results = []
    correct = 0
    for key, true_value in test_mapping_dict.items():
        search_results = search_method(key, top_n=1)
        predicted_value = search_results[0][2] if search_results else "Unmapped"
        is_correct = predicted_value == true_value
        if is_correct:
            correct += 1
        results.append(
            {
                "Key": key,
                "Predicted": predicted_value,
                "Ground Truth": true_value,
                "Correct": is_correct,
                "Score": f"{search_results[0][1]:.4f}",
            }
        )

    accuracy = correct / len(test_mapping_dict) * 100
    return pd.DataFrame(results), accuracy
