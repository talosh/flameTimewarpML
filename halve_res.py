import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Training script.')

    # Required argument
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    args = parser.parse_args()

    print (args.dataset_path)

if __name__ == "__main__":
    main()
