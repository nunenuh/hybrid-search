import string
from typing import List

import faiss
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

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

    def _build_index(self) -> faiss.IndexFlatL2:
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
            score = util.pytorch_cos_sim(query_embedding, corpus_embedding)
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

    def _query_embedding(self, query: str) -> np.ndarray:
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

    def _ranked_result(self, candidates: List[str], scores: List[float]) -> List[dict]:
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

    def bm25_search(self, query: str, top_n: int = 10) -> List[dict]:
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
        candidates = [self.corpus[doc_id] for doc_id in indices]
        scores = [scores[i] for i in indices]

        # Create results with scores
        ranked_results = self._ranked_result(candidates, scores)
        return ranked_results

    def transformer_search(self, query: str, top_n: int = 10) -> List[dict]:
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
        candidates = [self.corpus[doc_id] for doc_id in indices]

        # Create results with scores
        ranked_results = self._ranked_result(candidates, scores)

        return ranked_results

    def _rrf(
        self, rankings: List[List[int]], weights: List[float], k: int = 60
    ) -> dict:
        """
        Calculates the Reciprocal Rank Fusion (RRF) score with weights.

        Args:
            rankings (list of list of int): Rankings from different methods.
            weights (list of float): Weights for each ranking method.
            k (int): The constant for RRF.

        Returns:
            dict: Combined ranking scores.
        """
        rrf_scores = {}
        for weight, rank_list in zip(weights, rankings):
            for rank, doc_id in enumerate(rank_list):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + weight / (k + rank + 1)
        return rrf_scores

    def hybrid_search(
        self,
        query: str,
        top_n: int = 10,
        transformer_weight: float = 0.9,
        bm25_weight: float = 0.3,
    ) -> List[dict]:
        """
        Performs hybrid search combining BM25 and transformer-based search using weighted RRF.

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

        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_n]

        rankings = [tfr_indices.tolist(), bm25_indices.tolist()]
        weights = [transformer_weight, bm25_weight]
        rrf_scores = self._rrf(rankings, weights)

        combined_indices = list(rrf_scores.keys())
        combined_scores = [rrf_scores[doc_id] for doc_id in combined_indices]
        combined_candidates = [self.corpus[doc_id] for doc_id in combined_indices]

        ranked_results = self._ranked_result(combined_candidates, combined_scores)

        return ranked_results
