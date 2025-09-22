import os
import json
import requests
from pyDataverse.api import NativeApi, DataAccessApi
from tqdm import tqdm

# Configuration
API_TOKEN = "d67aad86-03da-49a0-a13e-169263f8c953"
DATAVERSE_SERVER = "https://dataverse.harvard.edu"
DATAVERSE_ALIAS = "harvard"
DATA_DIR = "data"

def setup_dataverse():
    """Set up the Dataverse API connection with timeout."""
    try:
        print("Connecting to Dataverse server...")
        api = NativeApi(DATAVERSE_SERVER, API_TOKEN)
        print("Connected to Dataverse API")
        
        # Test the connection
        print("Testing connection...")
        test = api.get_info_version()
        print(f"Dataverse version: {test}")
        
        data_api = DataAccessApi(DATAVERSE_SERVER, API_TOKEN)
        return api, data_api
    except Exception as e:
        print(f"Error setting up Dataverse API: {e}")
        if hasattr(e, 'response'):
            print(f"Status code: {e.response.status_code if hasattr(e.response, 'status_code') else 'N/A'}")
            try:
                print(f"Response: {e.response.text}")
            except:
                print("Could not get response text")
        return None, None

def create_dataset(api):
    """Create a new dataset with minimal metadata."""
    from pyDataverse.models import Dataset
    
    try:
        print("Creating dataset object...")
        dataset = Dataset()
        
        # Add required metadata
        print("Setting metadata...")
        metadata = {
            'title': 'A Multidimensional Evaluation Dataset for Trust in Conversational Agents',
            'author': [{'authorName': 'Gowtham', 'authorAffiliation': 'Independent Researcher'}],
            'datasetContact': [{
                'datasetContactEmail': 'your.email@example.com', 
                'datasetContactName': 'Gowtham'
            }],
            'dsDescription': [{
                'dsDescriptionValue': 'Dataset for research on trust in conversational AI.'
            }],
            'subject': ['Computer and Information Science'],
            'keyword': [
                {'keywordValue': 'Conversational AI'}, 
                {'keywordValue': 'Trust'}
            ]
        }
        
        print("Setting dataset metadata...")
        dataset.set(metadata)
        
        # Print the JSON to be sent (for debugging)
        dataset_json = dataset.json()
        print("Dataset JSON prepared. First 500 chars:", str(dataset_json)[:500] + "...")
        
        return dataset, dataset_json
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return None, None
    
    if not dataset or not dataset_json:
        print("Dataset preparation failed. Cannot create dataset.")
        return None
        
    try:
        print(f"\nCreating dataset in dataverse: {DATAVERSE_ALIAS}")
        print("Sending create dataset request...")
        
        # Create the dataset with a timeout
        resp = api.create_dataset(DATAVERSE_ALIAS, dataset_json, timeout=60)
        
        print(f"Response status code: {resp.status_code}")
        print(f"Response headers: {resp.headers}")
        
        if resp.status_code == 201:  # 201 Created
            try:
                resp_json = resp.json()
                print("Response JSON:", json.dumps(resp_json, indent=2))
                persistent_id = resp_json['data']['persistentId']
                print(f"Dataset created successfully with persistent ID: {persistent_id}")
                return persistent_id
            except Exception as e:
                print(f"Error parsing response: {e}")
                print(f"Raw response: {resp.text}")
                return None
        else:
            print(f"Failed to create dataset. Status code: {resp.status_code}")
            print(f"Response text: {resp.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("Error: Request timed out. The server took too long to respond.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Network/Request error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def upload_files(api, persistent_id):
    """Upload files to the dataset."""
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found.")
        return False
    
    # Get list of files to upload
    files_to_upload = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(('.json', '.csv', '.txt')):
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
            
            # Upload file
            with open(file_path, 'rb') as f:
                file_data = f.read()
                filename = os.path.basename(file_path)
                
                # Upload the file
                resp = api.upload_dataset_file(
                    persistent_id,
                    filename,
                    file_data,
                    description=f'Data file: {rel_path}'
                )
                
                if resp.status_code not in [200, 201]:
                    print(f"Error uploading {filename}: {resp.status_code} - {resp.text}")
                    
        except Exception as e:
            print(f"Error uploading {file_path}: {e}")
    
    return True

def publish_dataset(api, persistent_id):
    """Publish the dataset."""
    try:
        resp = api.publish_dataset(persistent_id, release_type='major')
        if resp.status_code == 200:
            print("Dataset published successfully!")
            return True
        else:
            print(f"Failed to publish dataset. Status code: {resp.status_code}")
            print(f"Response: {resp.text}")
            return False
    except Exception as e:
        print(f"Error publishing dataset: {e}")
        if hasattr(e, 'response'):
            print(f"Status code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        return False

def main():
    print("Starting Dataverse dataset creation and upload process...")
    
    # Set up API connection
    print("Setting up Dataverse API connection...")
    api, data_api = setup_dataverse()
    if not api or not data_api:
        print("Failed to set up Dataverse API. Exiting.")
        return
    
    # Step 1: Create dataset
    print("\nCreating dataset...")
    persistent_id = create_dataset(api)
    
    if not persistent_id:
        print("Failed to create dataset. Exiting.")
        return
    
    # Step 2: Upload files
    print("\nUploading files...")
    if not upload_files(api, persistent_id):
        print("File upload encountered errors. Dataset may be incomplete.")
    
    # Step 3: Publish dataset
    print("\nPublishing dataset...")
    if publish_dataset(api, persistent_id):
        print(f"\nDataset successfully published! View at: {DATAVERSE_SERVER}/dataset.xhtml?persistentId={persistent_id}")
    else:
        print("\nDataset created but not published due to errors.")

if __name__ == "__main__":
    main()
