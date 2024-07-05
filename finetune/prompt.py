mnli_3_shots_prompt_template = """Below is an instruction that describes a task, paired with some examples that provide further context. Write a response that appropriately completes the request.

### Instruction:
Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral).

### Examples:
Input: Premise: She smiled back. Hypothesis: She was so happy she couldn't stop smiling.
Response: Neutral.
Input: Premise: And to show just how fast Japan's new rulers were catching on, two punitive expeditions were launched against Korea and China in the grand manner of 19th-century gunboat diplomacy. Hypothesis: Japan's new rulers were catching on quickly.
Response: Entailment.
Input: Premise: Fun for adults and children. Hypothesis: Fun for only children.
Response: Contradiction.

### Input:
{}

### Response:
{}"""

bbh_3_shots_cot_prompt_template = """
Below is an instruction that describes a task, paired with some examples that provide further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Examples:
Input: {input1}
Response: {response1}
Input: {input2}
Response: {response2}
Input: {input3}
Response: {response3}

### Input:
{input}

### Response:
{response}
"""
bbh_all_prompts = {}
"""Load all the bbh prompts"""
import json
import csv
with open('../data_preparation/bbh/instructions.json', 'r') as file:
    data_instructions = json.load(file)
with open('../data_preparation/bbh/template.json', 'r') as file:
    data_examples = json.load(file)
for task_name in data_examples.keys():
    instruction = data_instructions[task_name]
    example_list = data_examples[task_name]
    input_prompt_list = [example[0] for example in example_list]
    output_prompt_list = ["Let's think step by step. "+example[1] for example in example_list]
    example = bbh_3_shots_cot_prompt_template.format(
        instruction=instruction,
        input1=input_prompt_list[0],
        input2=input_prompt_list[1],
        input3=input_prompt_list[2],
        response1=output_prompt_list[0],
        response2=output_prompt_list[1],
        response3=output_prompt_list[2],
        input="{input}",
        response="{response}"
    )
    bbh_all_prompts[task_name] = example
    
japanese_prompt_template ="""Below is an instruction that describes a task, paired with some examples that provide further context. Write a response that appropriately completes the request.

### Instruction:
Japanese-To-Python Generation
Pythonで1行のコードを生成し、StackOverflowの日本語の質問を解決してください。コメントや式は含めないでください。インポート文も不要です。
このタスクでは、入力は日本語のテキストで、変数名や操作が記述されています。出力は、そのタスクを達成するためのPythonの1行のコードです。コメントや式は含めないでください。インポート文も不要です。
                
Given a Japanese instruction, generate the according python code.

### Examples:
Input: スペースで区切られた入力`stdin`を変数に格納して表示する
Response: for line in stdin: a = line.rstrip().split(' ') print(a)
Input: リスト`word_list'内に出現する単語を数える
Response: Counter(word_list)
Input: tweepyインスタンス`api`を使い、文字列`word`を含んだツイートを検索し、結果をリストとして得る
Response: search = api.search(q=word)
Input: データベースの設定を表示する
Response: print(settings.DATABASES)
Input: ネストされているリスト`li`を見やすく表示する
Response: pprint.pprint(li)
Input: HTMLファイル'test.html'を開き、テキストオブジェクト'text'をutf-8で保存する
Response: f = open('test.html', 'w') f.write(text.encode('utf-8'))

### Input:
{}

### Response:
{}"""

squad_prompt_template ="""Below is an instruction that describes a task, paired with some examples that provide further context. Write a response that appropriately completes the request.

### Instruction:
Your task is to generate an answer to a natural question. In this task, the input is a string that consists of both a question and a context passage. The context is a descriptive passage related to the question and contains the answer. And the question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.

### Examples:
Input: "Question: What city did Super Bowl 50 take place in? Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50."
Response: "Santa Clara"
Input: "Question: What river runs through Warsaw? Context: Warsaw (Polish: Warszawa [varˈʂava] ( listen); see also other names) is the capital and largest city of Poland. It stands on the Vistula River in east-central Poland, roughly 260 kilometres (160 mi) from the Baltic Sea and 300 kilometres (190 mi) from the Carpathian Mountains. Its population is estimated at 1.740 million residents within a greater metropolitan area of 2.666 million residents, which makes Warsaw the 9th most-populous capital city in the European Union. The city limits cover 516.9 square kilometres (199.6 sq mi), while the metropolitan area covers 6,100.43 square kilometres (2,355.39 sq mi)."
Response: "Vistula River"
Input: "Question: The Ottoman empire controlled territory on three continents, Africa, Asia and which other? Context: The Ottoman Empire was an imperial state that lasted from 1299 to 1923. During the 16th and 17th centuries, in particular at the height of its power under the reign of Suleiman the Magnificent, the Ottoman Empire was a powerful multinational, multilingual empire controlling much of Southeast Europe, Western Asia, the Caucasus, North Africa, and the Horn of Africa. At the beginning of the 17th century the empire contained 32 provinces and numerous vassal states. Some of these were later absorbed into the empire, while others were granted various types of autonomy during the course of centuries."
Response: "Europe"

### Input:
{input}

### Response:
{response}"""
