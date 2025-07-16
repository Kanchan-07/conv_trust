import os
import json
from pathlib import Path
import shutil

# Define paths
base_path = Path(r'c:\Users\gowth\Downloads\conv_trust')
conversations_dir = base_path / 'data' / 'conversations'
metadata_dir = base_path / 'data' / 'metadata'

# Ensure metadata directory exists
metadata_dir.mkdir(parents=True, exist_ok=True)

print(f"Processing files in: {conversations_dir}")
print(f"Saving separated metadata to: {metadata_dir}")

processed_count = 0
deleted_count = 0
metadata_separated_count = 0
error_count = 0
no_metadata_skipped_count = 0

# Iterate through files in the conversations directory
for file_path in conversations_dir.glob('*.json'):
    processed_count += 1
    try:
        # Read the original content first
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if 'turns' key exists and is an empty list
        if 'turns' in data and isinstance(data['turns'], list) and not data['turns']:
            # Condition 1: Empty turns -> Delete file
            file_path.unlink()
            print(f"  - Deleted '{file_path.name}' (empty turns)")
            deleted_count += 1
        elif 'turns' in data: # turns is not empty
            # Condition 2: Non-empty turns -> Separate metadata if exists
            if 'metadata' in data:
                metadata = data['metadata']
                metadata_file_path = metadata_dir / file_path.name

                # Write the metadata to the metadata directory
                with open(metadata_file_path, 'w', encoding='utf-8') as mf:
                    json.dump(metadata, mf, indent=2)

                # Remove metadata from the original data
                del data['metadata']

                # Overwrite the original file without the metadata
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)

                print(f"  - Separated metadata from '{file_path.name}' to '{metadata_file_path}'")
                metadata_separated_count += 1
            else:
                # Non-empty turns, but no metadata key found
                print(f"  - Skipping '{file_path.name}' (no metadata key found, kept original)")
                no_metadata_skipped_count += 1
        else:
            # 'turns' key doesn't exist or is not a list
            print(f"  - Skipping '{file_path.name}' ('turns' key missing or not a list, kept original)")
            # Decide if these should be treated as errors or skipped

    except json.JSONDecodeError:
        print(f"  - Error: Could not decode JSON from '{file_path.name}'")
        error_count += 1
    except Exception as e:
        print(f"  - Error processing file '{file_path.name}': {e}")
        error_count += 1

print(f"\nProcessing complete.")
print(f"Total files processed: {processed_count}")
print(f"Files deleted (empty turns): {deleted_count}")
print(f"Metadata separated: {metadata_separated_count}")
print(f"Files skipped (no metadata key): {no_metadata_skipped_count}")
print(f"Errors encountered: {error_count}")
