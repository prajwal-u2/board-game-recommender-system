# Board Game Recommender System

A replication and extension of the ACM RecSys 2020 paper [*"Designing a Recommender System for Board Games"*](https://doi.org/10.1145/3341105.3375780) — built as part of CSCI 5123 (Recommender Systems) at the University of Minnesota.

**Team:** Tony Zhang, Prajwal Umesha, Shuo Jiang

---

## Overview

This project reproduces the recommender system pipeline from the original paper and extends it with additional deep learning architectures. We implemented and benchmarked 13+ algorithms across collaborative filtering, content-based, neural, and graph-based approaches — all evaluated on the BoardGameGeek (BGG) dataset.

Computationally intensive models (user-user and item-item CF) were run on the **Minnesota Supercomputing Institute (MSI)**. Deep learning models were trained on **Google Colab (NVIDIA L4 GPU)**.

---

## Dataset

**Source:** [BoardGameGeek](https://boardgamegeek.com/) ratings dataset

| Statistic | Paper | Our Dataset |
|---|---|---|
| Total ratings | 13M+ | **18.9M** |
| Unique games | 80,474 | **21,925** |
| Total users | 249,186 | **411,374** |
| Users after filtering (≥200 ratings) | ~N/A | **20,310** |
| Ratings used for modeling | ~N/A | **8.1M** |

**Features available per game:**
- Mechanics (boolean one-hot, 100+ mechanics)
- Categories (8 top-level: Thematic, Strategy, War, Family, CGS, Abstract, etc.)
- Themes, Publishers, Designers, Artists

**Preprocessing:**
- Filtered to users with at least 200 ratings (following the paper)
- 80/20 train/test split per user
- Ratings normalized per-user (mean-centered) for neural models
- Relevance threshold: test items with rating ≥ 7.0 treated as "liked" for evaluation

---

## Algorithms Implemented

### Baseline
| Model | Description |
|---|---|
| **Popularity** | Recommends globally most-rated games |

### Collaborative Filtering
| Model | Description |
|---|---|
| **User-User CF** | KNN-based, run on MSI HPC cluster |
| **Item-Item CF** | KNN-based, run on MSI HPC cluster |
| **Matrix Factorization** | SVD-based latent factor model |

### Neural / Deep Learning (PyTorch)
| Model | Description |
|---|---|
| **Classic Autoencoder (User-based)** | Encodes user interaction vectors |
| **Classic Autoencoder (Item-based)** | Encodes item interaction vectors |
| **CDAE — Denoising AE (User-based)** | Replication of paper's CDAE architecture |
| **Denoising AE (Item-based)** | Extension: item-view denoising AE |
| **Denoising AE + Gaussian Noise (User-based)** | Extension using Gaussian corruption instead of masking |
| **Variational AE — VAE (User-based)** | Latent-space sampling with KL divergence loss |
| **Variational AE — VAE (Item-based)** | Item-view VAE |
| **GNN** | Graph Neural Network over user-item bipartite graph |
| **Hybrid** | Combined autoencoder ensemble |

### Content-Based Filtering
| Model | Description |
|---|---|
| **Weighted KNN** | Inverse distance weighted KNN on game feature vectors |
| **IDF Content Filtering** | Smoothed TF-IDF over game mechanics/categories |

---

## Evaluation Framework

All models were evaluated with the following metrics, implemented from scratch to match paper definitions:

| Metric | Description |
|---|---|
| **Precision@K** | Fraction of top-K recommendations that are relevant |
| **Recall@K** | Fraction of relevant items captured in top-K |
| **NDCG@K** | Normalized Discounted Cumulative Gain — ranking quality |
| **MAP** | Mean Average Precision across users |
| **Diversity** | Average pairwise distance between recommended items |
| **Novelty** | How different recommendations are from a user's history |

K evaluated at: 5, 10, 20, 100.

---

## Results Summary

Best results per model (Precision@5 / NDCG@5):

| Model | Precision@5 | NDCG@5 | MAP |
|---|---|---|---|
| **GNN** | **0.371** | **0.391** | **0.097** |
| Item DAE (embedded) | 0.255 | 0.266 | 0.064 |
| Item Classic AE | 0.233 | 0.228 | 0.060 |
| Item Denoising AE | 0.230 | 0.230 | 0.059 |
| Matrix Factorization | 0.180 | 0.193 | 0.033 |
| User Classic AE | 0.176 | 0.192 | 0.033 |
| User DAE (Gaussian) | 0.143 | 0.162 | 0.029 |
| User CDAE | 0.126 | 0.140 | 0.026 |
| User VAE | 0.057 | 0.063 | 0.014 |
| KNN Content | 0.047 | 0.048 | 0.006 |
| IDF Content | 0.039 | 0.041 | 0.006 |
| Popularity Baseline | — | — | — |
| Item VAE | 0.012 | 0.028 | 0.002 |

**Key findings:**
- The GNN dramatically outperformed all other models, achieving nearly 2× the NDCG@5 of the next best.
- Item-based autoencoders consistently outperformed their user-based counterparts, likely because the item space (21K) is much more manageable than the user space.
- Denoising autoencoders outperformed classic autoencoders; Gaussian noise corruption slightly outperformed mask-based corruption.
- VAEs underperformed relative to classic AEs, likely due to the KL regularization pulling representations toward a unit Gaussian that doesn't suit sparse interaction data.
- Content-based models (KNN, IDF) performed significantly below collaborative approaches, reflecting the dominance of interaction signal.

---

## Hyperparameter Tuning

Extensive grid search was run over autoencoder architectures:

**User AE — Best config:**
- Hidden dim: 1024, Weight decay: 0.20, Optimizer: Adadelta, Epochs: 100, Batch: 256

**Item AE — Best config:**
- Hidden dim: 1024, Weight decay: 0.050, Optimizer: Adadelta, Epochs: 75–100, Batch: 256

Adadelta consistently outperformed Adam for autoencoder training on this dataset.

---

## Tech Stack

- **Python** — NumPy, Pandas, Scikit-learn, Scipy
- **PyTorch** — all neural models (autoencoders, VAEs, GNN)
- **Surprise** — collaborative filtering baselines
- **Google Colab** (L4 GPU) — neural model training
- **Minnesota Supercomputing Institute (MSI)** — user-user and item-item CF
- **Google Drive** — artifact storage and model checkpointing

---

## Paper Reference

Athanasios Tzelepis, Apostolos N. Papadopoulos, and Vasilis Katos. 2020.
*Designing a Recommender System for Board Games.*
In Proceedings of the 35th Annual ACM Symposium on Applied Computing (SAC '20).
https://doi.org/10.1145/3341105.3375780
