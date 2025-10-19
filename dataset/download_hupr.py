from huggingface_hub import snapshot_download

# Download the entire HuPR dataset
dataset_path = snapshot_download(
    repo_id="nirajpkini/HuPR",
    repo_type="dataset",
    local_dir="./data/HuPR",
    resume_download=True  # Resume if interrupted
)

print(f"Dataset downloaded to: {dataset_path}")