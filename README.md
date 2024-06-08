# Hybrid Search

## Overview
This project implements a solution for mapping various account names from multiple columns to preferred account names using a combination of traditional and advanced search techniques. The hybrid approach ensures accurate and efficient mapping by integrating keyword search with BM25 and similarity search with Sentence Transformers using FAISS.

## Problem Statement
The primary goal is to create an accurate and efficient mapping of account names using a hybrid search mechanism that combines BM25 and Sentence Transformers with FAISS. The project addresses the challenge of handling unmapped account names by leveraging advanced search techniques.

## Features
- **Initial Mapping**: Use a dictionary table to map account names from multiple columns to preferred account names.
- **Traditional Search**: Implement keyword search to retrieve account names based on the initial mapping.
- **Hybrid Search**: Combine BM25 and Sentence Transformers for improved accuracy in handling unmapped account names.
- **Evaluation**: Assess the search accuracy using confusion matrix and various metrics.

## Setup Instructions
### 1. Clone the Repository
Clone the repository to your local machine using the following command:
```bash
git clone https://github.com/yourusername/hybrid-search.git
cd hybrid-search
