from datasets import load_dataset

def load_data(max_rows):
    data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    data = data.to_pandas()
    data["id"] = data.index
    subset_data = data.head(max_rows)
    return subset_data
