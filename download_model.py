from transformers import AutoModel

def download_model():
    model_name = "jinaai/jina-embeddings-v2-base-en"  # Replace with your model name
    save_directory = "/model"  # This matches the WORKDIR in the Dockerfile

    print(f"Downloading model: {model_name}")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    print(f"Saving model to {save_directory}")
    model.save_pretrained(save_directory)
    
    print("Model downloaded and saved successfully")

if __name__ == "__main__":
    download_model()