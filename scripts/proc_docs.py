#!usr/bin/python3

from trec21_ct import process_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--document_path", help="Supply a path to the zip file of unprocessed xml documents.")
parser.add_argument("-w", "--write_file", help="Supply the name of the jsonl that will be created with the processed_docs")
parser.add_argument("-n", "--max_trials", help="Specify max number of trials to processs. Default is 1 million.", default=1000000)
args = parser.parse_args()


if __name__ == "__main__":
  process_data(zip_data=args.document_path, write_file=args.write_file, max_trials=max_trials)






