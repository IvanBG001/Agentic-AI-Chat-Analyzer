from app.utils.loader import DataLoader
from app.utils.cleaner import DataCleaner
from app.utils.transformer import DataTransformer

def main():
    # Step 1: Load Dataset
    loader = DataLoader(filepath="data/BiztelAI_DS_Dataset_V1.json")
    df = loader.load_data()
    print("✅ Loaded data:", df.shape)

    # Step 2: Clean Dataset
    cleaner = DataCleaner(df)
    df_clean = cleaner.clean_all()
    print("✅ Cleaned data:", df_clean.shape)

    # Step 3: Transform Dataset
    transformer = DataTransformer(df_clean)
    df_transformed = transformer.transform_all()
    print("✅ Transformed data. Sample:\n", df_transformed.head())

if __name__ == "__main__":
    main()
