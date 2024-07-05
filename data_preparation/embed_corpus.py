from sentence_transformers import SentenceTransformer
import json
import numpy as np
import pickle
from tqdm import tqdm
# Initialize the sentence-transformer model
model = SentenceTransformer('distiluse-base-multilingual-cased').to('cuda') # Example model, choose according to needs

def save_values():
    json_file_path = 'datasets_merged_flattened_final.jsonl'

    # Check if stored values already exist
    stored_values_file = 'stored_values.pkl'
    try:
        with open(stored_values_file, 'rb') as f:
            stored_values = pickle.load(f)
        print("Loaded stored values from disk.")
    except FileNotFoundError:
        print("Stored values not found. Processing JSON file.")
        stored_values = []
        with open(json_file_path, 'r') as file:
            for num, line in enumerate(file):
                data = json.loads(line)  # Assuming each line is a valid JSON object
                stored_values.append(data['value'])
        
        # Save stored values to disk
        with open(stored_values_file, 'wb') as f:
            pickle.dump(stored_values, f)
        print("Stored values saved to disk.")


def save_dict():
    json_file_path = 'datasets_merged_flattened_final.jsonl'

    # Check if stored values already exist
    stored_values_file = 'stored_dict.pkl'
    try:
        with open(stored_values_file, 'rb') as f:
            stored_values = pickle.load(f)
        print("Loaded stored values from disk.")
    except FileNotFoundError: 
        print("Stored values not found. Processing JSON file.")
        stored_values = {}
        with open(json_file_path, 'r') as file:
            for num, line in enumerate(file):
                data = json.loads(line)  # Assuming each line is a valid JSON object
                value = data['value']
                location = str(data['location'])
                column = data['column']
                if location not in stored_values.keys():
                    stored_values[location] = {column: value}
                else:
                    stored_values[location][column] = value
        # Save stored values to disk
        with open(stored_values_file, 'wb') as f:
            pickle.dump(stored_values, f)
        print("Stored values saved to disk.")

def save_embedding_info(dataset_index_path, json_file_path, embeddings_file_path_save):
        
    start = 0
    embeddings = None

    info_dict = {}
    with open(dataset_index_path, 'r') as f:
        info_dict = json.load(f)
    
    embedding_list = []
    name_to_description_dict = {}
    ids = []
    with open(json_file_path, 'r') as file:
        for num, line in enumerate(file):
            if num < start:
                continue
            data = json.loads(line)  # Assuming each line is a valid JSON object
            name = data['location']['dataset_id']
            ids.append(str(data['location']))
            name_dict = info_dict[name]
            description = name_dict['description']
            if name not in name_to_description_dict.keys():
                name_dict_embedding = model.encode([description], batch_size=1, show_progress_bar=True,convert_to_numpy=True) 
                name_to_description_dict[name] = name_dict_embedding
                if (num - start) == 0:
                    embeddings = name_dict_embedding
            if (num-start) > 0:
                embedding = name_to_description_dict[name]
                embedding_list.append(embedding)
            if (num-start) % 1000 == 0:
                if (num-start) > 0:
                    embedding = np.vstack(embedding_list)
                    embeddings = np.vstack((embeddings, embedding))
                with open(embeddings_file_path_save, "wb") as fOut:
                    pickle.dump({'ids': ids, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
                embedding_list = []
    
    # Save stored values to disk
    embedding = np.vstack(embedding_list)
    embeddings = np.vstack((embeddings, embedding))
    with open(embeddings_file_path_save, "wb") as fOut:
        pickle.dump({'ids': ids, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
      
if __name__=="__main__":
    save_values()
    save_dict()
    dataset_index_path = 'dataset_index.json'  # Path to dataset index
    json_file_path = 'datasets_merged_flattened_final.jsonl'  # Path to merged corpus
    embeddings_file_path_save = 'embeddings_dataset_final.pkl'  # Path to corpus embeddings
    save_embedding_info(dataset_index_path, json_file_path, embeddings_file_path_save)