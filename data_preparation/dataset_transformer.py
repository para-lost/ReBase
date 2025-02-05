from __future__ import annotations
import json
import anthropic
from tqdm import tqdm
import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import datasets
from api_tools import APIAgent
import csv
import ast
import re
import asyncio
from typing import List
from config import API_KEY, API_BASE, MODEL_NAME
import argparse

import os
os.environ['TIKTOKEN_CACHE_DIR'] = '../cache_dir'
                
def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None


def parse_json(
    response, required_keys: list, optional_keys: list
) -> dict | None:
    """Parse stuctured fields from the API response.

    Args:
        response: API response.
        required_keys: Required keys from the response
        optional_keys: Optional keys from the response

    Returns:
        If the API response is a valid JSON object and contains the
        required and optional keys then returns the
        final response as a Dictionary
        Else returns None.
    """
    # usage = response.choices[0]["usage"]["total_tokens"]
    response_text = response.choices[0]["message"]["content"]
    response_text = response_text.replace('{\n', '{')
    response_text = response_text.replace('}\n', '}')
    response_text = response_text.strip("```json\n")
    response_text = response_text.strip("```json")
    response_text = response_text.strip("\n```")
    response_text = response_text.strip("```")
    response_json = {}
    try:
        response_json = json.loads(response_text, strict=False)
    except json.decoder.JSONDecodeError:
        try: 
            response_json = {}
            response_text = str(response_text)
            response_text = response_text.strip("{").strip("}")

            if "\"input\":" in response_text:
                response_json['input'] = response_text.split("\"input\":")[1].split(",\n")[0]
                response_json['output'] = response_text.split("\"output\":")[1].split(",\n")[0]
            elif "'input':" in response_text:
                response_json['input'] = response_text.split("'input':")[1].split("'output'")[0]
                response_json['output'] = response_text.split("'output':")[1].split(",")[0]
            else:
                response_json['input'] = response_text.split("input")[1].split(",")[0]
                response_json['output'] = response_text.split("output")[1].split(",")[0]

        except:
            return None
        return None
    missing_keys = [key for key in required_keys if key not in response_json]
    if len(missing_keys) != 0:
        
        return None

    final_response = {}
    for key in required_keys + optional_keys:
        if key not in response_json:
            # This is an optional key, so exclude it from the final response.
            continue
        if type(response_json[key]) == str:
            final_response[key] = response_json[key].strip()
        else:
            final_response[key] = response_json[key]
    return final_response
    
def parse_json_azure(
    response, required_keys: list
) -> dict | None:
    """Parse stuctured fields from the API response.

    Args:
        response: API response.
        required_keys: Required keys from the response
        optional_keys: Optional keys from the response

    Returns:
        If the API response is a valid JSON object and contains the
        required and optional keys then returns the
        final response as a Dictionary
        Else returns None.
    """
    response = response.replace('{\n', '{')
    response = response.replace('}\n', '}')
    try:
        response_json = json.loads(response, strict=False)
    except json.decoder.JSONDecodeError:
        try: 
            response_json = {}
            response = str(response)
            response = response.strip("{").strip("}")
            response_json['input'] = response.split('"input":')[1].split(",\n")[0]
            response_json['output'] = response.split('"output":')[1].split(",\n")[0]
        except:
            return None
        

    missing_keys = [key for key in required_keys if key not in response_json]
    if len(missing_keys) != 0:
        return None

    final_response = {}
    for key in required_keys:
        if key not in response_json:
            # This is an optional key, so exclude it from the final response.
            continue
        if type(response_json[key]) == str:
            final_response[key] = response_json[key].strip()
        else:
            final_response[key] = response_json[key]
    return final_response
    
def str_to_dict(str_dict):
    try:
        # Convert the string to a dictionary
        actual_dict = ast.literal_eval(str_dict)
        return actual_dict
    except ValueError as e:
        # Handle the error if the string is not a valid dictionary
        print(f"Error converting string to dict: {e}")
        return None

    
def save_dataset_to_csv(dataset, file_path='transformed_dataset_japanese.csv'):
    """Save the dataset to a CSV file."""
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['input', 'output'])  # Writing header

        # Assuming dataset['train'] contains the data
        for row in dataset['train']:
            writer.writerow([row['input_col'], row['output_col']])
    print(f"Dataset saved to {file_path}")

def save_input_output_to_csv(input, output, file_path='transformed_dataset_japanese.csv'):
    """Save the dataset to a CSV file."""
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['input', 'output'])  # Writing header

        # Assuming dataset['train'] contains the data
        for input_col, output_col in zip(input, output):
            writer.writerow([input_col, output_col])
    print(f"Dataset saved to {file_path}")


TRANSFORM_DATA_PROMPT_VERSION = """
I would like you to create questions for a test. The directions for the test are:

```
{task_description}
```
The format should be in json like this:
{example}

Now I will provide you with a JSON file from a different dataset. Please create a question where the format and type of question is similar to the examples provided above, but the content is inspired by the example provided below.
You need to decide which part of the dataset to use.
{dataset_row}

Your response MUST be a JSON with exactly 2 fields: "input" and "output". 
Response (JSON ONLY):
"""  


FILTER_DATA_PROMPT_VERSION = """
You will be given a task description. Your task is to determine whether a data is fitful for this task.

# Instruction:
{task_description}

# Fitful Examples that meet the task's request:
{example}

Now, there is a new data. Your task is to determine whether this data is fitful for this task.
New Data:
{{
"input": "{input_data}",
"output": "{output_data}",
}}
Response (Yes or No):
"""  




EXAMPLE_TEMPLATE = """
```json
{{
"input": "{input_one}",
"output": "{output_one}",
}}

{{
"input": "{input_two}",
"output": "{output_two}",
}}

{{
"input": "{input_three}",
"output": "{output_three}",
}}
```
"""


def truncate_row(example_row: dict, max_length=250) -> str:
    """Truncate the row before displaying if it is too long."""
    truncated_row = {}
    for key in example_row.keys():
        curr_row = json.dumps(example_row[key])
        truncated_row[key] = (
            curr_row
            if len(curr_row) <= max_length - 3
            else curr_row[:max_length] + "..."
        )
    return json.dumps(truncated_row)



def construct_prompt_for_transform_data(
    task_description: str, dataset_row: dict, example: str, task_specific_instruction: str
) -> str:
    """Construct prompt for transform data."""
    return TRANSFORM_DATA_PROMPT_VERSION.format(
        task_description=task_description,
        dataset_row=truncate_row(dataset_row),
        example=example,
        task_specific_instruction=task_specific_instruction,
    )

def construct_prompt_for_filter_data(
    task_description: str, input_row: str, output_row:str, example: str, task_specific_instruction: str
) -> str:
    """Construct prompt for transform data."""
    return FILTER_DATA_PROMPT_VERSION.format(
        task_description=task_description,
        input_data=input_row,
        output_data=output_row,
        example=example,
        task_specific_instruction=task_specific_instruction,
    )



class PromptBasedDatasetTransformer():
    """Transform data based on a transform prompt."""

    def __init__(
        self,
        transform_prompt_fn: Callable[
            [str, dict, str, str], str
        ] = construct_prompt_for_transform_data,
    ):
        """Initialize the class."""
        self.transform_prompt_fn = transform_prompt_fn

    def make_dataset_from_samples(
        self,
        inputs: list[str],
        outputs: list[str],
    ) -> datasets.DatasetDict:
        """Given a list of inputs and outputs, make a dataset.

        This function takes in inputs and outputs, both as list of strings,
        and returns a DatasetDict object with a single split, "train". It has
        two columns, "input_col" and "output_col".


        Args:
            inputs: A list of inputs, each input is a string.
            outputs: A list of outputs, each output is a string.

        Returns:
            A DatasetDict object with a single split, "train". It has two
            columns, "input_col" and "output_col".
        """
        if len(inputs) <= 0 or len(inputs) != len(outputs):
            raise ValueError(f"Length of inputs and outputs must be >0 and equal. Cur length of inputs is: {len(inputs)}, cur length of outputs is: {len(outputs)}")

        dataset_dict = {}
        dataset_dict["train"] = datasets.Dataset.from_dict(
            {"input_col": inputs, "output_col": outputs}
        )
        return datasets.DatasetDict(dataset_dict)

    def transform_data(
        self,
        instruction,
        examples,
        task_specific_instruction,
        dataset,
        num_points_to_transform: int,
        file_path: str,
        use_azure=True
    ) -> datasets.DatasetDict:

        inputs = []
        outputs = []

        max_len = min(num_points_to_transform, len(dataset))
        len_count = 0
        transform_prompts = []
        for row in dataset:
            transform_prompt = self.transform_prompt_fn(
                instruction,
                row,
                examples,
                task_specific_instruction,
            )
            transform_prompts.append(transform_prompt)

            len_count += 1
            if len_count >= max_len:
                break
        batch_size = 20
        num_batches = int((len(transform_prompts) + batch_size - 1) / batch_size)
        def fetch_response(prompt):
            try:
                message = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    temperature=0.7,
                    system="You are a useful assistant",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return message.content[0].text
            except Exception as e:
                return str(e)
        async def generate_responses_openai(transform_prompts):
            default_api_agent = APIAgent(model_name=MODEL_NAME, api_base=API_BASE, api_key=API_KEY)
            responses = await default_api_agent.generate_batch_completion(
                transform_prompts,
                temperature=0,
                responses_per_request=1,
                requests_per_minute=20,
            )
            return responses
        def generate_responses(prompts: List[str]) -> List[str]:
            responses = []
            while len(prompts) > 0:
                un_completed_prompts = []
                with ThreadPoolExecutor(max_workers=200) as executor:
                    future_to_prompt = {executor.submit(fetch_response, prompt): prompt for prompt in prompts}
                    for future in tqdm(as_completed(future_to_prompt), total=len(prompts)):
                        try:
                            response = future.result()
                            if response != 'error':
                                responses.append(response)
                            else:
                                un_completed_prompts.append(future_to_prompt[future])
                                
                        except Exception as exc:
                            print(f'Prompt generated an exception: {exc}')
                prompts = un_completed_prompts
            return responses
        if use_azure:
            client = anthropic.Anthropic(
                api_key=API_KEY
            )
            all_response = []
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, len(transform_prompts))
                current_batch = transform_prompts[batch_start:batch_end]

                try:
                    responses = generate_responses(current_batch)
                except Exception as e:
                    print('error:', e)

                for response in responses:
                    print(f'response in transform = {response}')
                    try:
                        if use_azure:
                            extraction = parse_json_azure(response, ["input", "output"])
                        else:
                            extraction = parse_json(response, ["input", "output"], [])
                        print(f'extraction in transform = {extraction}')
                        if extraction is not None:
                            inputs.append(extraction["input"])
                            outputs.append(extraction["output"])
                        else:
                            print("error:\n"+response)
                    except Exception as e:
                        print('error:', e)
                        print('response:', response)

                all_response.extend(responses)
                
        else:
            try:
                loop = asyncio.get_event_loop()
                responses = loop.run_until_complete(generate_responses_openai(transform_prompts))
            except Exception as e:
                print('error:', e)

            for response in responses:
                print(f'response = {response}')
                
                try:
                    if use_azure:
                        extraction = parse_json_azure(response, ["input", "output"])
                    else:
                        extraction = parse_json(response, ["input", "output"], [])
                    print(f'extraction in transform = {extraction}')
                    if extraction is not None:
                        inputs.append(extraction["input"])
                        outputs.append(extraction["output"])
                except Exception as e:
                    print(f'error: {e}')

            if inputs and outputs:  # Ensure there's something to save
                save_input_output_to_csv(inputs, outputs, file_path)

        return self.make_dataset_from_samples(inputs, outputs)
    
    def filter_data(
        self,
        instruction,
        examples,
        task_specific_instruction,
        dataset,
        num_points_to_transform: int,
        file_path: str,
        use_azure=True
    ) -> datasets.DatasetDict:

        inputs = []
        outputs = []
        
        max_len = min(num_points_to_transform, len(dataset))
        len_count = 0
        transform_prompts = []
        
        for row in dataset:
            transform_prompt = self.transform_prompt_fn(
                instruction,
                row['input'],
                row['output'],
                examples,
                task_specific_instruction,
            )
            transform_prompts.append(transform_prompt)

            len_count += 1
            if len_count >= max_len:
                break
        batch_size = 1000
        num_batches = int((len(transform_prompts) + batch_size - 1) / batch_size)
        async def generate_responses(transform_prompts):
            responses = await api_tools.default_api_agent.generate_batch_completion(
                transform_prompts,
                temperature=0,
                responses_per_request=5,
                requests_per_minute=80,
            )
            return responses
        cur = 0
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(transform_prompts))
            current_batch = transform_prompts[batch_start:batch_end]
            original_current_batch = dataset[batch_start:batch_end]
            try:
                loop = asyncio.get_event_loop()
                responses = loop.run_until_complete(generate_responses(current_batch))
            except Exception as e:
                print(f"error: {e}")

            for response, row in zip(responses, original_current_batch):
                try:
                    response_text = response.choices[0]["message"]["content"]
                    answer = "yes" in response_text.lower()
                    if answer:
                        inputs.append(row["input"])
                        outputs.append(row["output"])
                    else:
                        cur += 1
                except Exception as e:
                    continue

            if inputs and outputs:  # Ensure there's something to save
                save_input_output_to_csv(inputs, outputs, file_path)
    
        if inputs and outputs:
            return self.make_dataset_from_samples(inputs, outputs)
        else:
            dataset_dict = {}
            dataset_dict["train"] = datasets.Dataset.from_dict(
                {"input_col": [""], "output_col": [""]}
            )
            return datasets.DatasetDict(dataset_dict)

    
def get_transformed_data_input_output(path='./selected_dataset', task_name='mconala', use_score=True, use_full_input=True, use_azure=True, instruction='', examples='', number=2000):
    if use_full_input and use_score:
        json_file_path = "./tasks/" + task_name + "/selected_dataset.json"
        json_file_path2 = "./tasks/" + task_name + "/retrieved_data_rank_score_dataset.json"
        data_list2 = load_json_file(json_file_path2)
    elif use_score:
        json_file_path = "./tasks/" + task_name + "/retrieved_data_rank_score.json"
    else:
        json_file_path = "./tasks/" + task_name + "/retrieved_data_rank2.json"
    data_list = load_json_file(json_file_path)
    if instruction == '':
        if task_name == 'mconala':
            instruction = """Japanese-To-Python Generation
                    Pythonで1行のコードを生成し、StackOverflowの日本語の質問を解決してください。コメントや式は含めないでください。インポート文も不要です。
                    このタスクでは、入力は日本語のテキストで、変数名や操作が記述されています。出力は、そのタスクを達成するためのPythonの1行のコードです。コメントや式は含めないでください。インポート文も不要です。
                    
                    Given a Japanese instruction, generate the according python code.
                    """
            examples = """
            ```json
            {
            \"input\": \"スペースで区切られた入力`stdin`を変数に格納して表示する\",
            \"output\": \"for line in stdin: a = line.rstrip().split(' ') print(a)\",
            }
            
            {
            \"input\": \"リスト`word_list'内に出現する単語を数える\",
            \"output\": \"Counter(word_list)\",
            }
            
            {
            \"input\": \"tweepyインスタンス`api`を使い、文字列`word`を含んだツイートを検索し、結果をリストとして得る\",
            \"output\": \"search = api.search(q=word)\",
            }
            
            {
            \"input\": \"データベースの設定を表示する\",
            \"output\": \"print(settings.DATABASES)\",
            }
            
            {        
            \"input\": \"ネストされているリスト`li`を見やすく表示する\",
            \"output\": \"pprint.pprint(li)\",
            }
            
            {
            \"input\": \"HTMLファイル'test.html'を開き、テキストオブジェクト'text'をutf-8で保存する\",
            \"output\": \"f = open('test.html', 'w') f.write(text.encode('utf-8'))\",
            }
                    """
            task_specific_instruction = """
            You need to make sure that the input is a Japanese instruction to write a Python code. the output is the corresponding Python code! 
            You may need to do modification with the provided data, eg translate english instruction into Japanese instruction.
            Don't try to be brief, make sure you output the complete information (no omission).
            """

        elif task_name == 'mnli':
            instruction = """
                Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral).
            """
            examples =  """
            {
            \"input\": \"Premise: She smiled back. Hypothesis: She was so happy she couldn't stop smiling.\",
            \"output\": \"The premise states that she smiled back, which does not show whether she couldn't stop smiling or not. so the answer is Neutral\",
            }
            
            {
            \"input\": \"Premise: And to show just how fast Japan's new rulers were catching on, two punitive expeditions were launched against Korea and China in the grand manner of 19th-century gunboat diplomacy. Hypothesis: Japan's new rulers were catching on quickly.\",
            \"output\": \"The premise states that the Japan's new rulers were catching on quickly and uses an example to demonstrate this. This entails the hypothesis that Japan's new rulers were catching on quickly. so the answer is Entailment\",
            }
            
            {
            \"input\": \"Premise: Fun for adults and children. Hypothesis: Fun for only children.\",
            \"output\": \"The premise indicates that the fun is for both adults and children, both contradicts with \"only children\", so the answer is Contradiction\",
            }
            """
            task_specific_instruction = """
            Try to use the Data Sample. 
            Don't try to be brief, make sure you output the complete information (no omission).
            """

        elif task_name == 'squad':
            instruction = """
            Your task is to generate an answer to a natural question. In this task, the input is a string that consists of both a question and a context passage. The context is a descriptive passage related to the question and contains the answer. And the question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.
            """
            examples =  """
            ```json
            {
            \"input\": \"Question: What city did Super Bowl 50 take place in? Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.\",
            \"output\": \"Santa Clara\",
            }
            
            {
            \"input\": \"Question: What river runs through Warsaw? Context: Warsaw (Polish: Warszawa [varˈʂava] ( listen); see also other names) is the capital and largest city of Poland. It stands on the Vistula River in east-central Poland, roughly 260 kilometres (160 mi) from the Baltic Sea and 300 kilometres (190 mi) from the Carpathian Mountains. Its population is estimated at 1.740 million residents within a greater metropolitan area of 2.666 million residents, which makes Warsaw the 9th most-populous capital city in the European Union. The city limits cover 516.9 square kilometres (199.6 sq mi), while the metropolitan area covers 6,100.43 square kilometres (2,355.39 sq mi).\",
            \"output\": \"Vistula River\",
            }
            
            {
            \"input\": \"Question: The Ottoman empire controlled territory on three continents, Africa, Asia and which other? Context: The Ottoman Empire was an imperial state that lasted from 1299 to 1923. During the 16th and 17th centuries, in particular at the height of its power under the reign of Suleiman the Magnificent, the Ottoman Empire was a powerful multinational, multilingual empire controlling much of Southeast Europe, Western Asia, the Caucasus, North Africa, and the Horn of Africa. At the beginning of the 17th century the empire contained 32 provinces and numerous vassal states. Some of these were later absorbed into the empire, while others were granted various types of autonomy during the course of centuries.\",
            \"output\": \"Europe\",
            }
            ```
            """
            task_specific_instruction = ''
    
    else:
        task_specific_instruction = ''
    dataset = []
    if use_full_input and use_score:
        for item, item2 in zip(data_list, data_list2):
            row = item
            if len(str(row)) > 4000:
                row_input = str(item2[2])
                row_output = str(item2[3])
                row = {"input": row_input, "output": row_output}
            dataset.append(row)
    else:
        for num, item in enumerate(data_list):
            if use_full_input:
                row = item
            elif use_score:
                row_input = str(item[2])
                row_output = str(item[3])
                row = {"input": row_input, "output": row_output}
            else:
                scores, rank = item[0], item[1]
                row_input = str(scores[3])
                row_output = str(scores[4])
                row = {"input": row_input, "output": row_output}
            dataset.append(row)
    prompt_transformer = PromptBasedDatasetTransformer()
    if use_full_input:
        save_path = "./tasks/"+task_name+"/transformed_data_score_use_full_row_dataset.csv"
    elif use_score:
        save_path = "./tasks/"+task_name+"/transformed_data_score.csv"
    else:
        save_path = "./tasks/"+task_name+"/transformed_data.csv"
    data = prompt_transformer.transform_data(instruction, examples, task_specific_instruction, dataset, number, save_path, use_azure=use_azure)
    save_dataset_to_csv(data, save_path)

    
def get_filtered_data_input_output(path='./selected_rows_rank', task_name='squad', use_score=True, use_full_input=True, use_azure=True, instruction='', examples='', number=2000):
    if use_full_input:
        file_path = "./tasks/"+task_name+"/transformed_data_score_use_full_row_dataset.csv"
    elif use_score:
        file_path = "./tasks/"+task_name+"/transformed_data_score.csv"
    else:
        file_path = "./tasks/"+task_name+"/transformed_data.csv"
    data_list = []
    with open(file_path, 'r') as f:
        csvFile = csv.DictReader(f)
        for lines in csvFile:
            row = {"input": lines['input'], "output": lines['output']}
            data_list.append(row)
    if instruction == '':
        if task_name == 'mconala':
            instruction = """Japanese-To-Python Generation
                    Pythonで1行のコードを生成し、StackOverflowの日本語の質問を解決してください。コメントや式は含めないでください。インポート文も不要です。
                    このタスクでは、入力は日本語のテキストで、変数名や操作が記述されています。出力は、そのタスクを達成するためのPythonの1行のコードです。コメントや式は含めないでください。インポート文も不要です。
                    
                    Given a Japanese instruction, generate the according python code.
                    """
            examples = """
            ```json
            {
            \"input\": \"スペースで区切られた入力`stdin`を変数に格納して表示する\",
            \"output\": \"for line in stdin: a = line.rstrip().split(' ') print(a)\",
            }
            
            {
            \"input\": \"リスト`word_list'内に出現する単語を数える\",
            \"output\": \"Counter(word_list)\",
            }
            
            {
            \"input\": \"tweepyインスタンス`api`を使い、文字列`word`を含んだツイートを検索し、結果をリストとして得る\",
            \"output\": \"search = api.search(q=word)\",
            }
            
            {
            \"input\": \"データベースの設定を表示する\",
            \"output\": \"print(settings.DATABASES)\",
            }
            
            {        
            \"input\": \"ネストされているリスト`li`を見やすく表示する\",
            \"output\": \"pprint.pprint(li)\",
            }
            
            {
            \"input\": \"HTMLファイル'test.html'を開き、テキストオブジェクト'text'をutf-8で保存する\",
            \"output\": \"f = open('test.html', 'w') f.write(text.encode('utf-8'))\",
            }
                    """
            task_specific_instruction = """
            You need to make sure that the input is a Japanese instruction to write a Python code. the output is the corresponding Python code! 
            You may need to do modification with the provided data, eg translate english instruction into Japanese instruction.
            Don't try to be brief, make sure you output the complete information (no omission).
            """

        elif task_name == 'mnli':
            instruction = """
                Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral).
            """
            examples =  """
            {
            \"input\": \"Premise: She smiled back. Hypothesis: She was so happy she couldn't stop smiling.\",
            \"output\": \"The premise states that she smiled back, which does not show whether she couldn't stop smiling or not. so the answer is Neutral\",
            }
            
            {
            \"input\": \"Premise: And to show just how fast Japan's new rulers were catching on, two punitive expeditions were launched against Korea and China in the grand manner of 19th-century gunboat diplomacy. Hypothesis: Japan's new rulers were catching on quickly.\",
            \"output\": \"The premise states that the Japan's new rulers were catching on quickly and uses an example to demonstrate this. This entails the hypothesis that Japan's new rulers were catching on quickly. so the answer is Entailment\",
            }
            
            {
            \"input\": \"Premise: Fun for adults and children. Hypothesis: Fun for only children.\",
            \"output\": \"The premise indicates that the fun is for both adults and children, both contradicts with \"only children\", so the answer is Contradiction\",
            }
            """
            task_specific_instruction = """
            Try to use the Data Sample. 
            Don't try to be brief, make sure you output the complete information (no omission).
            """

        elif task_name == 'squad':
            instruction = """
            Your task is to generate an answer to a natural question. In this task, the input is a string that consists of both a question and a context passage. The context is a descriptive passage related to the question and contains the answer. And the question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.
            """
            examples =  """
            ```json
            {
            \"input\": \"Question: What city did Super Bowl 50 take place in? Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.\",
            \"output\": \"Santa Clara\",
            }
            
            {
            \"input\": \"Question: What river runs through Warsaw? Context: Warsaw (Polish: Warszawa [varˈʂava] ( listen); see also other names) is the capital and largest city of Poland. It stands on the Vistula River in east-central Poland, roughly 260 kilometres (160 mi) from the Baltic Sea and 300 kilometres (190 mi) from the Carpathian Mountains. Its population is estimated at 1.740 million residents within a greater metropolitan area of 2.666 million residents, which makes Warsaw the 9th most-populous capital city in the European Union. The city limits cover 516.9 square kilometres (199.6 sq mi), while the metropolitan area covers 6,100.43 square kilometres (2,355.39 sq mi).\",
            \"output\": \"Vistula River\",
            }
            
            {
            \"input\": \"Question: The Ottoman empire controlled territory on three continents, Africa, Asia and which other? Context: The Ottoman Empire was an imperial state that lasted from 1299 to 1923. During the 16th and 17th centuries, in particular at the height of its power under the reign of Suleiman the Magnificent, the Ottoman Empire was a powerful multinational, multilingual empire controlling much of Southeast Europe, Western Asia, the Caucasus, North Africa, and the Horn of Africa. At the beginning of the 17th century the empire contained 32 provinces and numerous vassal states. Some of these were later absorbed into the empire, while others were granted various types of autonomy during the course of centuries.\",
            \"output\": \"Europe\",
            }
            ```
            """
            task_specific_instruction = ''
    
    else:
        task_specific_instruction = ''
    
    prompt_transformer = PromptBasedDatasetTransformer(construct_prompt_for_filter_data)
    if use_full_input:
        save_path = "./tasks/"+task_name+"/transformed_data_score_use_full_row_dataset_filtered.csv"
    elif use_score:
        save_path = "./tasks/"+task_name+"/transformed_data_score_filtered.csv"
    else:
        save_path = "./tasks/"+task_name+"/transformed_data_filtered.csv"
    data = prompt_transformer.filter_data(instruction, examples, task_specific_instruction, data_list, 1000, save_path, use_azure=use_azure)
    save_dataset_to_csv(data, save_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Retrieve data for a specific task.") 
    parser.add_argument("--task_name", type=str, help="Name of the task.")  
    args = parser.parse_args()
    TASK_NAME_LIST = [args.task_name]    
    use_filter = False
    use_azure = False
    
    for task_name in TASK_NAME_LIST:
        if "bbh" not in task_name:
            if use_filter:
                get_filtered_data_input_output(task_name=task_name, use_azure=use_azure, number=2000)
            else:
                get_transformed_data_input_output(task_name=task_name, use_azure=use_azure, number=2000)
        else:
            with open('./bbh/instructions.json', 'r') as file:
                data_instructions = json.load(file)
            with open('./bbh/template.json', 'r') as file:
                data_examples = json.load(file)
            for task_name in data_examples.keys():
                # with open("./tasks/bbh/"+task_name+"/transformed_data_score_use_full_row_dataset.csv", 'r') as file:
                #     reader = csv.reader(file)
                #     # Skip the header
                #     next(reader)
                #     # Count the rows
                #     row_count = sum(1 for row in reader)
                    
                instruction = data_instructions[task_name]
                example_list = data_examples[task_name]
                input_prompt_list = [example[0] for example in example_list]
                output_prompt_list = [example[1] for example in example_list]
                example = EXAMPLE_TEMPLATE.format(
                    input_one=input_prompt_list[0],
                    input_two=input_prompt_list[1],
                    input_three=input_prompt_list[2],
                    output_one=output_prompt_list[0],
                    output_two=output_prompt_list[1],
                    output_three=output_prompt_list[2],
                )
                if use_filter:
                    get_filtered_data_input_output(task_name="bbh/"+task_name, use_azure=use_azure, instruction=instruction, examples=example, number=2000)
                else:
                    get_transformed_data_input_output(task_name="bbh/"+task_name, use_azure=use_azure, instruction=instruction, examples=example, number=2000)            