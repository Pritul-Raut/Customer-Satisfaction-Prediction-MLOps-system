from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    
    path = "data/olist_customers_dataset.csv"
    train_pipeline(path)