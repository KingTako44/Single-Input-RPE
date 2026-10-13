import pandas as pd
import csv
from tqdm import trange
from RPE_w_example import Reverse_Prompt_Engineering

data_prompt = pd.read_csv("Dataset/testing_data_examples.csv")
data_answers = pd.read_csv("Dataset/testing_answers.csv")
list_prompt = list(data_prompt["prompt"])
list_category = list(data_prompt["category"])
list_example = list(data_prompt["example"])
list_answer = list(data_answers["answer"])
models = ["meta-llama/llama-3.3-70b-instruct", "openai/gpt-5.6-luna", "z-ai/glm-5.3-flash", "mistralai/ministral-3b-2512"]

for model in models:
    #record initial answers
    ori_answers = []
    #record final answer
    final_outputs = []
    model_name = model.split('/')[1]
    #initialize optimizer
    RPE = Reverse_Prompt_Engineering(X=3, N=3, model=model)
    n = len(list_prompt)
    
    with open(f'final_results/outputs_RPE_example_{model_name}.csv', 'w', encoding="utf-8", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['extracted_prompt', 'category'])
    with open(f'final_results/answers_RPE_example_{model_name}.csv', 'w', encoding="utf-8", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['answer', 'category'])
    
    print("Algorithm starts")
    for i in trange(n):
        p = list_prompt[i]
        e = list_example[i]
        a = list_answer[i]
        RPE.reset(X=3, N=3)
        RPE.generate_answers(prompt=p, example=e, answers=a, n=1)
        output = RPE(ini_prompts=None, ini_scores=None, ini_total_scores=None, ini_answers=None)
        with open(f'final_results/outputs_RPE_example_{model_name}.csv', 'a', encoding="utf-8", newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow([output, list_category[i]])
        with open(f'final_results/answers_RPE_example_{model_name}.csv', 'a', encoding="utf-8", newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow([RPE.answers, list_category[i]])
