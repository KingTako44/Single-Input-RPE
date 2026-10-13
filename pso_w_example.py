import openai
from openai import APITimeoutError
import re
import numpy as np
import evaluate

class Reverse_Prompt_Engineering():
    def __init__(self, X=1, N=1, model="gpt-3.5-turbo"):
        '''
        Initialize the optimization process
        parameter:
          X: number of iteration for optimization
          N: number of candidate prompts to keep at each optimization step
          p: probability to add words from answers to children (mutation probability) #not using
          tag_model: #not using
        '''
        self.X = X
        self.N = N
        self.llm = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="YOUR OPENROUTER API KEY HERE",
        )
        self.answers = None
        self.model = model
        self.candidate_prompts = []
        self.candidate_scores = []
        self.candidate_total_scores = []
        self.best_score = None
        self.best_prompt = None
        self.candidate_answers = []
        self.parent_candidates = []
        self.ori_prompt = None
        self.rouge = evaluate.load('rouge')
        self.current_prompts = []
        self.current_answers = [None] * self.N
        self.example = None

    def reset(self, X=1, N=1):
        self.X = X
        self.N = N
        self.answers = None
        
        self.candidate_prompts = []
        self.candidate_scores = []
        self.candidate_total_scores = []
        self.best_score = None
        self.best_prompt = None
        self.candidate_answers = []
        self.parent_candidates = []
        self.current_prompts = []
        self.current_answers = [None] * self.N
        
    def __call__(self, ini_prompts=None, ini_scores=None, ini_total_scores=None, ini_answers=None):
        """
        Recover prompt from answers
        """
        #generate prompt based on answers
        self.initial_prompt_generation(ini_prompts)
        #print("-----------------------------------generated initial prompt-----------------------------------")
        #evaluation
        if ini_scores and ini_total_scores and ini_answers:
            self.candidate_scores, self.candidate_total_scores, self.candidate_answers = ini_scores, ini_total_scores, ini_answers
        else:
            self.candidate_scores, self.candidate_total_scores, self.candidate_answers = self.evaluation(self.candidate_prompts)
        self.best_score = self.candidate_scores[0]
        self.best_prompt = self.candidate_prompts[0]
        #print("-----------------------------------evaluated initial prompt-----------------------------------")

        
        for i in range(self.X):
            #crossover
            children_candidates = self.cross_mutation()
            #print(f"-----------------------------------finished mutuation {i}-----------------------------------")

            #evaluation
            
            children_scores, children_total_scores, children_answers = self.evaluation(children_candidates)
            #print(f"-----------------------------------finished evaluation {i}-----------------------------------")

            #replacement
            self.replacement(children_candidates, children_scores, children_answers, children_total_scores)
            #print(f"-----------------------------------finished replacement {i}-----------------------------------")

        return self.best_prompt
        


    def chat(self, messages, model="openai/gpt-4o-mini", n=1, t=0.5):
        """
        define chat function
        """
        while True:
            try:
                response = self.llm.chat.completions.create(
                    model=model,
                    messages=messages,
                    n=n,
                    temperature=t,
                    timeout = 120
                )
                if (response is None) or (response.choices is None):
                    print('Returned None response, retrying')
                    print('Message:', messages)
                else:
                    break
            except APITimeoutError:
                print('API request timed out, retrying')
        text_response = response.choices[0].message.content

        
        return text_response

    def generate_answers(self, prompt=None, example=None, answers=None, n:int=1, t=0.5):
        """
        Generate answers we want to infer from. Create tag dict for all answers
        parameter:
          prompt: the original prompt we use to generate answers
          answers: optional, if provided, just copy this list to self.answers
          n: number of answers to generate for searching original prompt
        """
        if answers:
            self.answers = answers
        else:
            message = [{"role":"user", "content":prompt}]
            self.answers = self.chat(message, model="openai/gpt-4o-mini", n=n, t=t)

        #print(self.answers)
        self.ori_prompt = prompt
        self.example = example
        

    def initial_prompt_generation(self, ini_prompts=None):
        if ini_prompts:
            for p in ini_prompts:
                self.candidate_prompts.append(p)
        else:
            for i in range(self.N):
                prompt_0 = "The answer above is generated bypassing a prompt through. Based on the answer provided above, speculate the underlying prompt. "+\
                    "Give only one prompt in your response. Return the prompt beginning with <PRO> and end with </PRO>."
                message_0 = [{"role":"system", "content":"As a smart detective, you can infer the underlying prompt besed on outputs of language model."}] + \
                        [{"role":"user", "content":"Answer: "+str(self.answers)}] + \
                        [{"role":"user", "content":prompt_0}] + \
                        [{"role":"system", "content":"Think step by step and return the prompt beginning with <PRO> and end with </PRO>."}]
                candidate_prompt = self.chat(message_0, model=self.model, n=self.N, t=1)
                while candidate_prompt is None:
                    #print('regenerating initial prompt')
                    candidate_prompt = self.chat(message_0, model=self.model, n=self.N, t=1)
                splited_prompt = re.split('<PRO>|</PRO>', candidate_prompt)
                #print(candidate_prompt)
                #if the format is not correct, regenerate the prompt until the format is correct
                while len(splited_prompt) != 3:
                    #print('regenerating')
                    prompt = self.chat(message_0, model=self.model, n=1)
                    splited_prompt = re.split('<PRO>|</PRO>', prompt)
                    #print(splited_prompt)
                self.candidate_prompts.append(splited_prompt[1])
                self.current_prompts.append(splited_prompt[1])

    def evaluation(self, prompt_list = []):
        """
        Evaluate each prompt candidate in prompt_list
        """
        total_score_list = []
        final_score_list = []
        candidate_answers_list = []
        n = len(prompt_list)

        for j in range(n):
            
            message_prompt = [{"role":"user", "content":prompt_list[j]}]
            answer_e = self.chat(message_prompt, model=self.model, t=0)
            while answer_e is None:
                answer_e = self.chat(message_prompt, model=self.model, t=0)
            score_list = []
            self.current_answers[j] = answer_e
            #print(answer_e)
                
            predictions = [answer_e]
            references = [self.answers]
            results = self.rouge.compute(predictions=predictions,
                        references=references)
            score = results["rouge1"]
            score_list.append(float(score))
                
            final_score = (np.mean(score_list)+max(score_list))/2
            #final_score = np.mean(score_list)
            #final_score = max(score_list)
            total_score_list.append(score_list)
            final_score_list.append(final_score)
            candidate_answers_list.append(answer_e)
            
        return final_score_list, total_score_list, candidate_answers_list

    def cross_mutation(self):
        '''
        For each candidate prompt, get difference between candidate answer and original answers
        summarize that
        ask llm to give advise of revising prompt
        do the revise
        '''
        children_candidates = []
        
        for i in range(self.N):
            own_best_prompt = self.candidate_prompts[i]
            own_current_answer = self.current_answers[i]

            #difference between answers
            prompt_e = "You should answer concisely with no redundant information. Given two answers below, how is the candidate answer different from "+\
            "the reference answer? If there is no significant difference, you can answer 'There is no difference' in those exact words. Do not respond with nothing."
            message_e = [{"role":"user", "content": prompt_e}]+\
                        [{"role":"user", "content": "Candidate Answer: "+own_current_answer}]+\
                        [{"role":"user", "content": "Reference Answer: "+self.answers}]
            suggest_e = self.chat(message_e, model=self.model)
            while suggest_e is None:
                #print('regenerating')
                suggest_e = self.chat(message_e, model=self.model)
            #print(suggest_e)
            #print(f"-----------------------------------finished mutation step 1 (iteration {i})-----------------------------------")

            #generate advice to revise prompt
            prompt_t = "According to the 3 example prompts above, please provide a summary of the traits shared by the prompts. "+\
                        "Keep in mind that these example prompts are only approximations of an original prompt. Answer concisely."
            message_t = [{"role":"user", "content": "Example Prompt 1: " + self.example}] + \
                        [{"role":"user", "content": "Example Prompt 2: " + self.best_prompt}] + \
                        [{"role":"user", "content": "Example Prompt 3: " + own_best_prompt}] + \
                        [{"role":"user", "content": prompt_t}]
            traits_t = self.chat(message_t, model=self.model)
            while traits_t is None:
                #print('regenerating')
                traits_t = self.chat(message_t, model=self.model)
            #print(traits_t)
            #print(f"-----------------------------------finished mutation step 2 (iteration {i})-----------------------------------")

            #generate advice to revise prompt
            prompt_p = "According to the key differences and traits above, how should you change the prompt that "+\
                        "generated the candidate answer to make the candidate answer more similar to the reference answer? "+\
                        "If there is no difference, do not give any extra advice. Answer concisely."
            message_p = [{"role":"user", "content": "Differences: " + suggest_e}] + \
                        [{"role":"user", "content": "Prompt Traits " + traits_t}] + \
                        [{"role":"user", "content": prompt_p}] 
            difference_p = self.chat(message_p, model=self.model)
            while difference_p is None:
                #print('regenerating')
                difference_p = self.chat(message_p, model=self.model)
            #print(difference_p)
            #print(f"-----------------------------------finished mutation step 3 (iteration {i})-----------------------------------")

            #revise the prompt
            prompt_r = "Using the suggestions above to revise the following prompt." 
            message_r = [{"role":"user", "content": "Suggestion: " + difference_p}] + \
                        [{"role":"user", "content": prompt_r}] + \
                        [{"role":"user", "content": "Prompt: " + self.current_prompts[i]}] + \
                        [{"role":"user", "content": "The prompt should be concise. Return the prompt beginning with <PRO> and end with </PRO>."}]
            child_r = self.chat(message_r, model=self.model)
            while child_r is None:
                #print('regenerating')
                child_r = self.chat(message_r, model=self.model)
            #print(child_r)
            #print(f"-----------------------------------finished mutation step 4 (iteration {i})-----------------------------------")

            split_child = re.findall(r"<PRO>(.*?)</PRO>", child_r, re.DOTALL)
            while len(split_child) != 1 or split_child[0] == '':
                child_r = self.chat(message_r, model=self.model)
                split_child = re.findall(r"<PRO>(.*?)</PRO>", child_r, re.DOTALL)
                #print('regenerated')
            #print('Revised Prompt: ' + split_child[0])
            children_candidates.extend(split_child)
            self.current_prompts[i] = split_child[0]
            
        return children_candidates
        
    def replacement(self, children_prompts, children_scores, children_answers, children_total_scores):
        for i in range(self.N):
            child_score = children_scores[i]
            #print(f'New Score for Particle {i}: {child_score}')
            #print(f'Current Best Score for Particle {i}: {self.candidate_scores[i]}')
            #print(f'Current Best Score Overall: {self.best_score}')
            if child_score > self.best_score or self.best_score is None:
                #print(f'Overall Best Score Updated: {self.best_score} -> {child_score}')
                #print(f'Overall Best Prompt Updated: {self.best_prompt} -> {children_prompts[i]}')
                self.best_score = child_score
                self.best_prompt = children_prompts[i]
            if child_score > self.candidate_scores[i]:
                #print(f'Personal Best Score Updated for Particle {i}: {self.candidate_scores[i]} -> {child_score}')
                #print(f'Personal Best Prompt Updated for Particle {i}: {self.candidate_prompts[i]} -> {children_prompts[i]}')
                self.candidate_prompts[i] = children_prompts[i]
                self.candidate_answers[i] = children_answers[i]
                self.candidate_total_scores[i] = children_total_scores[i]
                self.candidate_scores[i] = child_score
            #print('--------------------------------------------------------------')
