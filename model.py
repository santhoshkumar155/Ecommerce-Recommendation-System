import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# 🔹 Step 1: Load dataset
df = pd.read_csv("data/amazon.csv")

print("Original Data:")
print(df.head())

# 🔹 Step 2: Show all column names (VERY IMPORTANT)
print("\nColumns in dataset:")
print(df.columns)

# 🔹 Step 3: Rename columns safely
df = df.rename(columns={
    'reviews.username': 'user_id',
    'asins': 'product_id',
    'reviews.rating': 'rating',
    'reviews.rating.1': 'rating'   # fallback if exists
})

# 🔹 Step 4: Check if required columns exist
required_cols = ['user_id', 'product_id', 'rating']
for col in required_cols:
    if col not in df.columns:
        print(f"❌ Column missing: {col}")
        exit()

# 🔹 Step 5: Select required columns
df = df[['user_id', 'product_id', 'rating']]

# 🔹 Step 6: Clean data
df.dropna(inplace=True)
df['rating'] = df['rating'].astype(float)

print("\nCleaned Data:")
print(df.head())

# 🔹 Step 7: Load into Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df, reader)

# 🔹 Step 8: Split data
trainset, testset = train_test_split(data, test_size=0.2)

# 🔹 Step 9: Train model
model = SVD()
model.fit(trainset)

# 🔹 Step 10: Evaluate model
predictions = model.test(testset)
print("\nModel Accuracy:")
accuracy.rmse(predictions)

# 🔹 Step 11: Recommendation function
def recommend_products(user_id, n=5):
    all_products = df['product_id'].unique()
    rated_products = df[df['user_id'] == user_id]['product_id'].values
    
    predictions = []
    
    for product in all_products:
        if product not in rated_products:
            pred = model.predict(user_id, product)
            predictions.append((product, pred.est))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return predictions[:n]

# 🔹 Step 12: Test recommendation
sample_user = df['user_id'].iloc[0]

print("\nRecommendations for user:", sample_user)
print(recommend_products(sample_user))