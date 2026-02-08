from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id="BharathonAI/tourism-newplan-adoption-prediction"        # the target repo
repo_type="space"
space_sdk="docker",  # Because we're using a Docker backend
private=False  # Make it private if needed

# api = HfApi(token=os.getenv("HF_TOKEN"))
# api.upload_folder(
#     folder_path="deployment",     # the local folder containing your files
#     repo_id="BharathonAI/tourism-newplan-adoption-prediction",          # the target repo
#     repo_type="space",                      # dataset, model, or space
#     path_in_repo="",                          # optional: subfolder path inside the repo
# )

try:
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type,space_sdk=space_sdk,private =private)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="deployment",
    repo_id=repo_id,
    repo_type=repo_type,
)
