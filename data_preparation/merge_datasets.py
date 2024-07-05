import pyarrow.parquet as pq
import pyarrow as pa
from datasets import load_dataset
import json
import time
import datasets
import pandas as pd
import traceback
import random
import os
import threading
from collections.abc import MutableMapping

LAST_PROCESSED_NAME = "last_processed_dataset_flatten.txt"

def fetch_first_row_with_timeout(dataset: datasets.Dataset, timeout: int = 30): 
    """
    Fetch the first row of a dataset within a specified timeout period.

    Args:
        dataset: The dataset from which to fetch the first row.
        timeout: The maximum time in seconds towait for
                 fetching the row (default is 30 seconds).

    Returns:
        dict or None: The first row of the dataset as a dictionary,
                      or None if the operation times out.
    """

    def fetch_sample_row(container):
        try:
            container.append(next(iter(dataset)))
        except Exception as e:
            container.append(e)

    result_container = []
    fetch_thread = threading.Thread(target=fetch_sample_row, args=(result_container,))
    fetch_thread.start()
    fetch_thread.join(timeout)

    if fetch_thread.is_alive() or result_container[0] is None:
        # Operation took too long or failed
        return None

    return result_container[0] if isinstance(result_container[0], dict) else None


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    """
    Flatten the sample rows of streaming dataset.

    Streaming Datasets from HF don't inherently have the flatten function.

    Args:
        d (MutableMapping): The dictionary to flatten.
        parent_key (str): The base key string to use for the flattened keys.
        sep (str): Separator used between nested keys (default is '.').

    Returns:
        dict: A flattened dictionary with no nested structures.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def replace_duplicate_columns(original_dataset_columns: list):
    """Replace duplicate column names in a dataset after flattening.

    Args:
        original_dataset_columns: List of original column names in the dataset.

    Returns:
        tuple: A tuple containing two elements:
                1. A list of new column names with duplicates handled.
                2. A dictionary mapping original column names to new column names.
    """
    columns_mapping: dict[str, str] = {}
    new_columns = []
    counter: dict[str, int] = {}
    # convert flattened columns like answer.text -> answer_text
    for col in original_dataset_columns:
        new_col = col.replace(".", "_")
        if new_col in columns_mapping.values():
            counter[new_col] = counter.get(new_col, 0) + 1
            new_col = f"{new_col}_{counter[new_col]}"
        columns_mapping[col] = new_col
        new_columns.append(new_col)
    return new_columns, columns_mapping
    

def save_last_processed_dataset(dataset_name):
    with open(LAST_PROCESSED_NAME, "w") as file:
        file.write(dataset_name)


def get_last_processed_dataset():
    try:
        with open(LAST_PROCESSED_NAME, "r") as file:
            print("The process starts from last processed dataset.")
            return file.read()
    except FileNotFoundError:
        print("The process starts from the beginning.")
        return None  


def get_label_names(dataset):
    """
    Check if the 'features' attribute has a 'label' field with names
    """
    label_names = None
    if 'label' in dataset.features and hasattr(dataset.features['label'], 'names'):
        label_names = dataset.features['label'].names
    return label_names


def get_dataset_length(dataset_name, config_name):
    """
    Attempt to get the dataset length from its metadata
    """
    try:
        dataset_info = datasets.get_dataset_infos(dataset_name)[config_name]
        return dataset_info.splits['train'].num_examples
    except Exception as e:
        print(f"Could not determine length for {dataset_name} with config {config_name}: {e}")
        return None
            

def process_row(row, dataset_id, config_id, row_id, label_names):
    processed_data = []
    row = flatten_dict(row)
    # Creating a list of dictionaries, each with 'value' and 'location' keys
    for column_name, value in row.items():
        # If column is label, convert the number to the actual content.
        if ('id' in column_name or 'label' in column_name) and label_names:
            value = label_names[value] if value in range(len(label_names)) else value

        processed_data.append({
            'column': column_name,
            'value': str(value), 
            'location': {'dataset_id': dataset_id, 'config_id':config_id, 'row_id': row_id}
        })
    
    return processed_data

def write_to_jsonl(data, file_path):
    with open(file_path, 'a') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')


def process_dataset_config(dataset_name, config_name, file_path, dataset_id, chunk_size=10000, max_samples=1000):
    dataset_length = get_dataset_length(dataset_name, config_name)
    # Taking too long to load, neglect for now
    if dataset_length > 50000:
        return
    dataset = datasets.load_dataset(
        dataset_name,
        config_name,
        split="train",
        streaming=True,
        download_mode="force_redownload",
        cache_dir="./.cache",
    )
    sample_rows = fetch_first_row_with_timeout(dataset, timeout=30)
    if not sample_rows:
        return
    sample_rows = flatten_dict(sample_rows)
    if any(
        "ImageFile" in sample_rows[key].__class__.__name__
        or "DateTime" in sample_rows[key].__class__.__name__
        for key in sample_rows
    ):
        return
    columns, columns_mapping = replace_duplicate_columns(list(sample_rows.keys()))
    label_names = get_label_names(dataset)
    fields = [
        pa.field('column', pa.string()),  # New field for the column name
        pa.field('value', pa.string()),
        pa.field('location', pa.struct([
            pa.field('dataset_id', pa.string()), 
            pa.field('config_id', pa.string()), 
            pa.field('row_id', pa.string())
        ]))
    ]
    table_schema = pa.schema(fields)
    processed_chunk = []
    # Keep track of total rows processed
    total_rows = 0  

    if dataset_length is not None:
        k = max(1, dataset_length // max_samples)
    else:
        k = 1  # or any other default value you choose
    start_time = time.time()
    for row_index, row in enumerate(dataset):
        if row_index % k == 0:
            processed_chunk.extend(process_row(row, dataset_id, config_name, str(total_rows), label_names))
            total_rows += 1
            # Check if the processed_chunk size has reached the chunk_size
            if len(processed_chunk) >= chunk_size:
                table = pa.Table.from_pandas(pd.DataFrame(processed_chunk), schema=table_schema)
                write_to_jsonl(processed_chunk, file_path)
                processed_chunk = []  # Reset chunk
            end_time = time.time()
        if total_rows >= max_samples or abs(end_time-start_time) > 300 :
            break

    # Process any remaining rows in the last chunk
    if processed_chunk:
        table = pa.Table.from_pandas(pd.DataFrame(processed_chunk), schema=table_schema)
        write_to_jsonl(processed_chunk, file_path)
         
def process_dataset(dataset_name, parquet_file, dataset_id, chunk_size=10000):
    config_names = datasets.get_dataset_config_names(dataset_name)
    all_configs = {}
    start_time = time.time()
    for config_name in config_names:
        if "train" not in datasets.get_dataset_split_names(dataset_name, config_name):
            continue
        process_dataset_config(dataset_name, config_name, parquet_file, dataset_id, chunk_size=10000)
        end_time = time.time()
        if abs(end_time - start_time) > 300:
            return


def get_dataset_names(dataset_index_path):
    with open(dataset_index_path, 'r') as json_file:
        return json.load(json_file)
    dataset_names = [details['name'] for details in json_file.values()]
    return dataset_names

def process_dataset_with_error_handling(dataset_name, parquet_file, dataset_id, chunk_size=10000):
    try:
        process_dataset(dataset_name, parquet_file, dataset_id, chunk_size)
    except Exception as e:
        print(f"An error occurred while processing {dataset_name}: {e}")
        # Print error's stack trace
        traceback.print_exc()  

        
if __name__ == "__main__":
    last_processed = get_last_processed_dataset()
    start_processing = False if last_processed else True
    
    # TODO: change dataset_index_path
    dataset_index_path = './dataset_index.json'
    dataset_names = get_dataset_names(dataset_index_path)  
    for name in dataset_names:
        if start_processing or name == last_processed:
            start_processing = True  
            # Start or resume processing
            if name == last_processed:
                continue
            start_time = time.time()
            process_dataset_with_error_handling(name, 'datasets_merged_flattened_final.jsonl', name)
            end_time = time.time()
            print(f"Finished processing {name}, Process took {(end_time-start_time)} seconds")
            save_last_processed_dataset(name)