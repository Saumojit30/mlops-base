import os
import argparse
from huggingface_hub import HfApi, create_repo

def upload_to_huggingface(repo_id, file_path, version_tag=None):
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

    dest_filename = os.path.basename(file_path)
    print(f"Uploading {file_path} as '{dest_filename}'...")
    
    try:
        commit_msg = f"Upload model weights: {dest_filename}"
        if version_tag:
            commit_msg += f" (Release {version_tag})"

        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=dest_filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg
        )
        print(f"\n🎉 File '{dest_filename}' uploaded successfully!")

        # Create tag on Hugging Face if provided
        if version_tag:
            try:
                api.create_tag(
                    repo_id=repo_id,
                    tag=version_tag,
                    repo_type="model"
                )
                print(f"🏷️  Hugging Face Version Tag '{version_tag}' created successfully!")
            except Exception as tag_err:
                print(f"⚠️ Could not create tag '{version_tag}' (it may already exist): {tag_err}")

        print(f"\nView your model here: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"\n❌ Error during upload: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a trained model to Hugging Face with Version Tagging")
    parser.add_argument(
        "--repo_id", 
        type=str, 
        default="Jit0777/california-housing-model",
        help="The Hugging Face repository ID (e.g., username/repo-name)"
    )
    parser.add_argument(
        "--file_path", 
        type=str, 
        default="models/xgb_model.joblib",
        help="Path to the local model file"
    )
    parser.add_argument(
        "--version", 
        type=str, 
        default="v2.0",
        help="Version tag to assign on Hugging Face (e.g. v2.0)"
    )
    
    args = parser.parse_args()
    upload_to_huggingface(args.repo_id, args.file_path, args.version)
