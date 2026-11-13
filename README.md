# Category-Partitioned FAISS Product Similarity Engine

**A Streamlit demonstration of category-aware nearest-neighbor product recommendations.**

This project builds an in-memory FAISS index for each product category. It imputes missing numeric values, transforms review counts with `log1p`, standardizes price/rating/review features, and searches for products with similar feature vectors.

## Features

- Loads a product catalog from `cleaned_product_catalog.csv`.
- Imputes missing numeric values using the median.
- Applies a logarithmic transform to review counts.
- Standardizes numeric features with scikit-learn.
- Builds a separate FAISS `IndexFlatL2` for each category.
- Retrieves the nearest products for a selected category and preference vector.
- Presents the workflow through a Streamlit interface.

## Technology

Python, Streamlit, pandas, NumPy, scikit-learn, and FAISS CPU.

## Requirements

- Python 3.9+ (choose a version compatible with the installed FAISS wheel)
- pip
- The product catalog CSV included in this repository

## Installation

```bash
git clone https://github.com/nyohleonard-nlktech/faiss-vector-product-engine.git
cd faiss-vector-product-engine

python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Streamlit prints a local URL (normally `http://localhost:8501`). Open it in your browser.

## How recommendations are calculated

1. The app loads the CSV into a pandas DataFrame.
2. Missing values in the numeric input columns are imputed.
3. Review counts are transformed using `log1p` to reduce long-tail scale effects.
4. Price, rating, and transformed review count are standardized.
5. Each category receives its own FAISS L2 index.
6. User preferences are transformed using the fitted scaler and queried against the selected category index.
7. Results are displayed with vector distance and a derived similarity percentage.

The displayed similarity percentage is a UI transformation of L2 distance, not a calibrated probability that a user will like a product.

## Data expectations

The current implementation expects the catalog to include these fields:

- `Category_ID`
- `Category_Name`
- `Product_Price`
- `Product_Rating`
- `Number_of_Reviews`

Check the CSV headers before replacing the sample dataset. Numeric fields should be parseable as numbers.

## Project structure

```text
.
├── app.py
├── cleaned_product_catalog.csv
├── requirements.txt
└── README.md
```

## Limitations and next steps

- Indexes are built in memory when the app starts; they are not persisted between restarts.
- The current engine uses numeric metadata rather than text embeddings.
- L2 distance is used for ranking; it is not interchangeable with cosine similarity.
- Add data validation, reproducible evaluation, and tests for empty categories and missing fields before production use.
- For large catalogs, benchmark approximate FAISS indexes and index persistence.

## License

No license file is currently present. All rights are reserved unless an explicit license is added.
