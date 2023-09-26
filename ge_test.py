from preprocessing.output_json import handle_data
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="prepare datasets")
    parser.add_argument('-f', '--file_path', dest='file_path', required=True)
    args = parser.parse_args()
    file_path = args.file_path
    handle_data(file_path)