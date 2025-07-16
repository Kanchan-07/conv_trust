import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

# Configuration
API_TOKEN = "d67aad86-03da-49a0-a13e-169263f8c953"
DATAVERSE_SERVER = "https://dataverse.harvard.edu"
DATAVERSE_ALIAS = "harvard"  # The alias for the main Harvard Dataverse
DATA_DIR = "data"

# Headers for API requests
headers = {
    "X-Dataverse-key": API_TOKEN,
    "Content-Type": "application/json"
}

def create_dataset():
    """Create a new dataset in Dataverse with the provided metadata."""
    url = f"{DATAVERSE_SERVER}/api/dataverses/{DATAVERSE_ALIAS}/datasets"
    
    # Minimal required metadata for dataset creation
    dataset_metadata = {
        "datasetVersion": {
            "metadataBlocks": {
                "citation": {
                    "fields": [
                        # Title (required)
                        {"typeName": "title", "value": "A Multidimensional Evaluation Dataset for Trust in Conversational Agents"},
                        
                        # Author (required)
                        {"typeName": "author", "value": [
                            {
                                "authorName": {"typeName": "authorName", "value": "Gowtham"},
                                "authorAffiliation": {"typeName": "authorAffiliation", "value": "Independent Researcher"}
                            }
                        ]},
                        
                        # Dataset Contact (required)
                        {"typeName": "datasetContact", "value": [
                            {
                                "datasetContactName": {"typeName": "datasetContactName", "value": "Gowtham"},
                                "datasetContactEmail": {"typeName": "datasetContactEmail", "value": "your.email@example.com"}
                            }
                        ]},
                        
                        # Description (required)
                        {"typeName": "dsDescription", "value": [
                            {
                                "dsDescriptionValue": {"typeName": "dsDescriptionValue", "value": "Dataset for research on trust in conversational AI."}
                            }
                        ]},
                        
                        # Subject (required)
                        {"typeName": "subject", "value": [
                            "Computer and Information Science"
                        ]},
                        
                        # Keywords (required)
                        {"typeName": "keyword", "value": [
                            {"keywordValue": "Conversational AI"},
                            {"keywordValue": "Trust"}
                        ]}
                    ]
                }
            },
            "termsOfUse": "CC0"
        }
    }
    
    try:
        print("Sending request to:", url)
        print("Request headers:", json.dumps(headers, indent=2))
        print("Request payload:", json.dumps(dataset_metadata, indent=2))
        
        response = requests.post(url, headers=headers, json=dataset_metadata, timeout=30)
        
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {response.headers}")
        print(f"Response text: {response.text}")
        
        response.raise_for_status()
        
        try:
            dataset_data = response.json()
            print("JSON Response:", json.dumps(dataset_data, indent=2))
            if 'data' in dataset_data and 'persistentId' in dataset_data['data']:
                persistent_id = dataset_data['data']['persistentId']
                print(f"Dataset created successfully with persistent ID: {persistent_id}")
                return persistent_id
            else:
                print("Unexpected response format. Missing 'data' or 'persistentId' in response.")
                print("Full response:", dataset_data)
                return None
                
        except json.JSONDecodeError as je:
            print(f"Failed to decode JSON response: {je}")
            print(f"Raw response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error creating dataset: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response headers: {e.response.headers}")
            print(f"Response text: {e.response.text}")
        return None

def upload_files(dataset_pid):
    """Upload files from the data directory to the dataset."""
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found.")
        return False
    
    # Get list of files to upload
    files_to_upload = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.json') or file.endswith('.csv') or file.endswith('.txt'):
                files_to_upload.append(os.path.join(root, file))
    
    if not files_to_upload:
        print(f"No JSON, CSV, or TXT files found in {DATA_DIR}")
        return False
    
    print(f"Found {len(files_to_upload)} files to upload...")
    
    # Upload each file
    for file_path in tqdm(files_to_upload, desc="Uploading files"):
        try:
            # Get relative path for description
            rel_path = os.path.relpath(file_path, DATA_DIR)
            
            # Prepare file data
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f)}
                data = {
                    'jsonData': json.dumps({
                        'description': f'Data file: {rel_path}',
                        'categories': ['Data']
                    })
                }
                
                # Upload file
                url = f"{DATAVERSE_SERVER}/api/datasets/:persistentId/add"
                params = {'persistentId': dataset_pid}
                headers_upload = {
                    'X-Dataverse-key': API_TOKEN
                }
                
                response = requests.post(url, params=params, headers=headers_upload, data=data, files=files)
                response.raise_for_status()
                
        except Exception as e:
            print(f"Error uploading {file_path}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
    
    return True

def publish_dataset(dataset_pid):
    """Publish the dataset."""
    try:
        url = f"{DATAVERSE_SERVER}/api/datasets/:persistentId/actions/:publish"
        params = {
            'persistentId': dataset_pid,
            'type': 'major'
        }
        
        response = requests.post(url, headers=headers, params=params)
        response.raise_for_status()
        print("Dataset published successfully!")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error publishing dataset: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return False

def main():
    print("Starting Dataverse dataset creation and upload process...")
    
    # Step 1: Create dataset
    print("Creating dataset...")
    dataset_pid = create_dataset()
    
    if not dataset_pid:
        print("Failed to create dataset. Exiting.")
        return
    
    print(f"Dataset created with persistent ID: {dataset_pid}")
    
    # Step 2: Upload files
    print("\nStarting file upload...")
    if not upload_files(dataset_pid):
        print("File upload encountered errors. Dataset may be incomplete.")
    
    # Step 3: Publish dataset
    print("\nPublishing dataset...")
    if publish_dataset(dataset_pid):
        print(f"\nDataset successfully published! View at: {DATAVERSE_SERVER}/dataset.xhtml?persistentId={dataset_pid}")
    else:
        print("\nDataset created but not published due to errors.")

if __name__ == "__main__":
    main()
