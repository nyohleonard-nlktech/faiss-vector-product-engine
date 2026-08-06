# Category-Partitioned FAISS Vector Similarity Search Engine

An end-to-end vector search recommendation engine that indexes e-commerce product catalogs into normalized vector spaces using **FAISS (Facebook AI Similarity Search)** and performs $K$-Nearest Neighbor ($K$-NN) queries over standardized feature distributions.

---

## 📌 Architectural Overview

Standard Euclidean distance metrics fail when applied directly to raw, un-scaled feature spaces containing mixed dynamic ranges (e.g., Price ranging from 10 to 500 vs. Rating ranging from 1.0 to 5.0 vs. Reviews ranging from 0 to 10000). This application solves scale dominance and long-tailed distributions through a multi-stage preprocessing pipeline prior to indexing candidate vectors into isolated FAISS instances per product category.
