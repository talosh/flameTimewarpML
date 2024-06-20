import os
import sys
import shutil

def process_files(source_directory, destination_directory):
    # List all files in the source directory
    files = os.listdir(source_directory)
    
    for file in files:
        if file.endswith('.exr'):
            # Extract the number from the filename
            file_number = file.split('_')[0]
            file_suffix = file.split('_')[1].split('.')[0]

            # Create a new folder based on the file number
            new_folder = os.path.join(destination_directory, file_number)
            os.makedirs(new_folder, exist_ok=True)

            # Determine the new filename based on the suffix
            if file_suffix == 'incoming':
                new_filename = '001.exr'
            elif file_suffix == 'target':
                new_filename = '002.exr'
            elif file_suffix == 'outgoing':
                new_filename = '003.exr'
            else:
                continue  # Skip any files that do not match the specified suffixes

            # Copy the file to the new folder with the new name
            source_file = os.path.join(source_directory, file)
            destination_file = os.path.join(new_folder, new_filename)
            shutil.copy2(source_file, destination_file)

if __name__ == "__main__":
    source_directory = sys.argv[1]  # Replace with the path to your source directory
    destination_directory = sys.argv[2]  # Replace with the path to your destination directory
    process_files(source_directory, destination_directory)
