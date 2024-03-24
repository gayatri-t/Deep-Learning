# Deep-learning-Homework 2

The codes are based on PyTorch 2.0, Python 3.6.

## Running the Code

To run the code, follow these steps:

1. **Download Sequence-to-Sequence Model:**
   - Download the Sequence-to-Sequence model from the provided link and save it in the root directory.

2. **Download Datasets:**
   - Download the specific dataset, including `testing_data`, `training_data`, `testing_label.json`, and `training_label.json`.
   - Update the hardcoded paths in `Test.py`, `Train.py`, and `Model.py` with the paths to these files.

3. **Update Paths in `hw2_seq2seq.sh`:**
   - Change the paths in `hw2_seq2seq.sh` to match your folder-specific paths.
   - Example: Replace `$testing_data` with the actual path to testing data and `$path_to_save_output.txt` with the desired output file path.
```bash
# Example paths update in hw2_seq2seq.sh
Python Test.py $"your/testing_data/path" $="your/path/to/save/output.txt"
