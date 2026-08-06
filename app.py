import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import faiss

# Set page config
st.set_page_config(page_title="Vector Similarity Search Engine", layout="wide")

# Custom CSS with unsafe_allow_html=True
st.markdown(r"""
    <style>
    .main { background-color: #0e1117; color: #e0e0e0; }
    .author-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 20px;
    }
    .highlight-text { color: #60a5fa; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

class ProductRecommenderEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.raw_df = None
        self.processed_df = None
        self.indices = {}
        self.category_maps = {}
        self.feature_cols = ['Product_Price', 'Product_Rating', 'Number_of_Reviews_log']

    def fit(self, df: pd.DataFrame):
        self.raw_df = df.copy()
        df_work = df.copy()
        numeric_cols = ['Product_Price', 'Product_Rating', 'Number_of_Reviews']
        df_work[numeric_cols] = self.imputer.fit_transform(df_work[numeric_cols])
        df_work['Number_of_Reviews_log'] = np.log1p(df_work['Number_of_Reviews'])
        
        scaled_features = self.scaler.fit_transform(df_work[self.feature_cols])
        for idx, col in enumerate(self.feature_cols):
            df_work[f"z_{col}"] = scaled_features[:, idx]
            
        self.processed_df = df_work
        unique_categories = df_work['Category_ID'].dropna().unique()
        for cat_id in unique_categories:
            cat_mask = (df_work['Category_ID'] == cat_id)
            cat_sub = df_work[cat_mask].copy()
            vectors = cat_sub[[f"z_{c}" for c in self.feature_cols]].values.astype('float32').copy()
            
            d = vectors.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(vectors)
            self.indices[cat_id] = index
            self.category_maps[cat_id] = cat_sub.index.tolist()

    def recommend_by_preferences(self, category_id: float, price: float, rating: float, reviews_count: int, k: int = 5) -> pd.DataFrame:
        if category_id not in self.indices:
            return pd.DataFrame()
        query_raw = pd.DataFrame([[price, rating, np.log1p(reviews_count)]], columns=self.feature_cols)
        query_vec = self.scaler.transform(query_raw).astype('float32')
        
        index = self.indices[category_id]
        cat_indices = self.category_maps[category_id]
        search_k = min(k, len(cat_indices))
        
        distances, positions = index.search(query_vec, search_k)
        matched_dataset_indices = [cat_indices[p] for p in positions[0] if p != -1]
        results = self.processed_df.loc[matched_dataset_indices].copy()
        results['Vector_Distance'] = distances[0][:len(matched_dataset_indices)]
        results['Similarity (%)'] = np.clip((1.0 / (1.0 + results['Vector_Distance'])) * 100.0, 0, 100)
        
        display_cols = ['Category_Name', 'Category_ID', 'Product_Price', 'Product_Rating', 'Number_of_Reviews', 'Vector_Distance', 'Similarity (%)']
        return results[[c for c in display_cols if c in results.columns]]

@st.cache_resource
def load_engine():
    df = pd.read_csv("cleaned_product_catalog.csv")
    engine = ProductRecommenderEngine()
    engine.fit(df)
    return engine

engine = load_engine()

# Sidebar with unsafe_allow_html=True
with st.sidebar:
    st.title("Control Panel")
    st.markdown("""
    <div class="author-card">
        <h4 style="margin:0; color:#60a5fa;">Engineered By</h4>
        <h3 style="margin:0; color:#ffffff;">Nyoh Leonard</h3>
        <p style="font-size: 0.85rem; color:#9ca3af; margin-top:4px;">Computer Engineering | AI & Vector Systems</p>
    </div>
    """, unsafe_allow_html=True)

st.title("⚡ Vector Similarity Product Engine")

tab1, tab2 = st.tabs(["🚀 Search", "🔬 Deep Technical Architecture"])

with tab1:
    st.markdown("### Interactive Query")
    cat_options = {name: cid for cid, name in zip(engine.processed_df['Category_ID'].unique(), engine.processed_df['Category_Name'].unique())}
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_cat_name = st.selectbox("Category:", options=list(cat_options.keys()))
        selected_cat_id = cat_options[selected_cat_name]
        k_val = st.slider("Top-K:", 1, 20, 5)
    with col2:
        price_val = st.slider("Price ($):", 10.0, 500.0, 250.0, 5.0)
        rating_val = st.slider("Rating:", 1.0, 5.0, 4.5, 0.1)
    with col3:
        reviews_val = st.number_input("Reviews:", 0, 10000, 500)

    if st.button("Search Vector Engine", type="primary"):
        results = engine.recommend_by_preferences(selected_cat_id, price_val, rating_val, reviews_val, k=k_val)
        st.dataframe(results.style.format({
            'Product_Price': '${:.2f}', 'Product_Rating': '{:.2f}',
            'Number_of_Reviews': '{:,.0f}', 'Vector_Distance': '{:.4f}', 'Similarity (%)': '{:.2f}%'
        }), use_container_width=True)

with tab2:
    st.header("🔬 Deep Technical Architecture")
    st.markdown(r"""
    ### Mathematical Pipeline
    1. **Log Normalization:** $x_{\text{rev\_log}} = \ln(1 + x_{\text{rev}})$
    2. **$Z$-Score Standardization:** $z_i = \frac{x_i - \mu_i}{\sigma_i}$
    3. **FAISS Search:** Index type `IndexFlatL2` calculating $d_{L2}(\mathbf{q}, \mathbf{x}) = \sqrt{\sum (q_j - x_j)^2}$.
    """)
