import os
import argparse
from huggingface_hub import HfApi, create_repo

def upload_to_huggingface(repo_id, file_path):
    print(f"Preparing to upload to Hugging Face Hub: {repo_id}")
    
    api = HfApi()
    
    try:
        # Create the repository if it doesn't already exist
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✅ Repository '{repo_id}' confirmed/created on Hugging Face.")
    except Exception as e:
        print("\n❌ ERROR: Authentication required or repository creation failed.")
        print("Please ensure you run 'hf auth login' in your terminal first!")
        print(f"Details: {e}")
        return

    print(f"Uploading {file_path}...")
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload trained model weights"
        )
        print("\n🎉 Upload complete!")
        print(f"View your model here: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"\n❌ Error during upload: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a trained model to Hugging Face")
    # For open source users, they must provide their own repo ID. 
    # But for YOUR convenience, it defaults to your repository if you don't type anything!
    parser.add_argument(
        "--repo_id", 
        type=str, 
        default="Jit0777/california-housing-model",
        help="The Hugging Face repository ID (e.g., username/repo-name)"
    )
    parser.add_argument(
        "--file_path", 
        type=str, 
        default="models/rf_model.joblib",
        help="Path to the local model file"
    )
    
    args = parser.parse_args()
    upload_to_huggingface(args.repo_id, args.file_path)
