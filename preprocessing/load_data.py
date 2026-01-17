import pandas as pd

def load_dataset(file_path, column_names):
    """
    Load the dataset in chunks to handle large file sizes.

    :param file_path: Path to the dataset file.
    :param column_names: List of column names for the dataset.
    :return: Pandas DataFrame containing the dataset.
    """
    chunks = []
    chunk_size = 10000  # Number of rows per chunk

    try:
        for chunk in pd.read_csv(file_path, names=column_names, chunksize=chunk_size):
            chunks.append(chunk)

        # Combine all chunks into a single DataFrame
        dataset = pd.concat(chunks, ignore_index=True)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Define column names based on NSL-KDD dataset documentation
    column_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
        "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "label", "difficulty_level"
    ]

    train_file = "KDDTrain+.txt"
    test_file = "KDDTest+.txt"

    train_data = load_dataset(train_file, column_names)
    test_data = load_dataset(test_file, column_names)

    if train_data is not None:
        print("Training data loaded successfully.")
        print(train_data.head())

    if test_data is not None:
        print("Test data loaded successfully.")
        print(test_data.head())
