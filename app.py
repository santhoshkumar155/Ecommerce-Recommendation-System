import streamlit as st
from model import recommend_products, df

st.title("🛒 E-commerce Recommendation System")

user_list = df['user_id'].unique()

user = st.selectbox("Select User", user_list)

if st.button("Get Recommendations"):
    results = recommend_products(user)
    
    st.write("### Recommended Products:")
    
    for product, score in results:
        st.write(f"Product: {product} ⭐ Score: {score:.2f}")