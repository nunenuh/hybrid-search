import string
import nltk
import numpy as np
import faiss
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')
nltk.download('stopwords')

class HybridSearch:
    def __init__(
        self, 
        base_mapping: dict, 
        transformer_model: str = 'yseop/roberta-base-finance-hypernym-identification'
    ):
        self.base_mapping = base_mapping
        self.corpus = list(base_mapping.keys())
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.model = SentenceTransformer(transformer_model)

        self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=False)
        self.fidx = self._initialize_faiss_index()

    def _initialize_faiss_index(self) -> faiss.IndexFlatL2:
        d = self.corpus_embeddings.shape[1]  # dimension of embeddings
        index = faiss.IndexFlatL2(d)
        index.add(np.array(self.corpus_embeddings))
        return index

    def _tokenize(self, text: str) -> list:
        text = text.lower()  # Convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        tokens = word_tokenize(text)  # Tokenize
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]  # Stem words
        return tokens

    def bm25_search(self, query: str, top_n: int = 10):
        # Preprocess the query
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get the top N candidates from BM25
        top_n_indices = np.argsort(bm25_scores)[::-1][:top_n]
        top_n_candidates = [self.corpus[i] for i in top_n_indices]
        top_n_bm25_scores = [bm25_scores[i] for i in top_n_indices]

        # Create results with scores
        ranked_results = [(top_n_candidates[i], top_n_bm25_scores[i], self.base_mapping[top_n_candidates[i]]) for i in range(len(top_n_candidates))]
        return ranked_results

    def transformer_search(self, query: str, top_n: int = 10):
        # Calculate Sentence Transformers Similarity using FAISS
        query_embedding = self.model.encode(query, convert_to_tensor=False).reshape(1, -1)
        _, faiss_indices = self.fidx.search(query_embedding, top_n)
        faiss_indices = faiss_indices.flatten()

        # Get the corresponding scores from FAISS
        faiss_scores = [util.pytorch_cos_sim(query_embedding, np.expand_dims(self.corpus_embeddings[i], axis=0)).numpy().flatten()[0] for i in faiss_indices]

        # Get the top N candidates
        top_n_candidates = [self.corpus[i] for i in faiss_indices]

        # Create results with scores
        ranked_results = [(top_n_candidates[i], faiss_scores[i], self.base_mapping[top_n_candidates[i]]) for i in range(len(top_n_candidates))]
        return ranked_results
        
    def hybrid_search(self, query: str, top_n: int = 10, transformer_weight: float = 0.9, bm25_weight: float = 0.3):
        # Calculate Sentence Transformers Similarity
        query_embedding = self.model.encode(query, convert_to_tensor=False).reshape(1, -1)
        _, faiss_indices = self.fidx.search(query_embedding, top_n)
        faiss_indices = faiss_indices.flatten()

        # Get the top N candidates from Sentence Transformers
        top_n_candidates = [self.corpus[i] for i in faiss_indices]
        transformer_scores = [util.pytorch_cos_sim(query_embedding, np.expand_dims(self.corpus_embeddings[i], axis=0)).numpy().flatten()[0] for i in faiss_indices]

        # Preprocess the query for BM25
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get BM25 scores for the top N candidates
        bm25_top_n_scores = [bm25_scores[self.corpus.index(candidate)] for candidate in top_n_candidates]

        # Normalize BM25 scores and transformer scores
        if np.max(bm25_top_n_scores) != np.min(bm25_top_n_scores):
            bm25_top_n_scores = (bm25_top_n_scores - np.min(bm25_top_n_scores)) / (np.max(bm25_top_n_scores) - np.min(bm25_top_n_scores))
        else:
            bm25_top_n_scores = np.array(bm25_top_n_scores)

        if np.max(transformer_scores) != np.min(transformer_scores):
            transformer_scores = (transformer_scores - np.min(transformer_scores)) / (np.max(transformer_scores) - np.min(transformer_scores))
        else:
            transformer_scores = np.array(transformer_scores)

        # Combine BM25 and Sentence Transformers scores (weighted sum)
        # Refined Scoring
        agreement_factor = np.multiply(bm25_top_n_scores, transformer_scores)  # Emphasize agreement
        combined_scores = bm25_weight * bm25_top_n_scores + transformer_weight * transformer_scores + agreement_factor
        # combined_scores = bm25_weight * np.array(bm25_top_n_scores) + transformer_weight * np.array(transformer_scores)

        # Rank results based on combined scores
        ranked_indices = np.argsort(combined_scores)[::-1]
        ranked_results = [(top_n_candidates[i], combined_scores[i], self.base_mapping[top_n_candidates[i]]) for i in ranked_indices]
        
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
        results.append({
            "Key": key, 
            "Predicted": predicted_value, 
            "Ground Truth": true_value, 
            "Correct": is_correct,
            "Score": f"{search_results[0][1]:.4f}",
        })
    
    accuracy = correct / len(test_mapping_dict) * 100
    return pd.DataFrame(results), accuracy