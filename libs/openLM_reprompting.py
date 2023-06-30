import sys
sys.path.insert(0, '../../')
from libs.text_prompts import text_prompts

import re
import argparse

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaTokenizer

import numpy as np
import utils.const as const
import utils.sys_const as sys_const

SUPPORTED_MODEL = [const.T5_NAME, const.GPT2_NAME, const.LLAMA_NAME]

def get_z_prompts_gpt2(dataset_name, question_prompts):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
    # question_prompts = text_prompts[dataset_name]['question_openLM']
    prompt_template = text_prompts[dataset_name]['prompt_template']
    z_prompts = []
    for prompt in question_prompts:
        answers = []
        for q in prompt:
            inputs = tokenizer(q, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=10)
            resp_visible_differences = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split(q)[-1].strip().rstrip().replace('\xa0', ' ')
            resp_visible_differences = resp_visible_differences.split(',')[0].split('.')[0].lower()
            print(q, resp_visible_differences)
            if len(prompt_template) > 0 and prompt_template in resp_visible_differences:
                resp_visible_differences = resp_visible_differences.split(prompt_template)[-1]
            for label_ in text_prompts[dataset_name]['labels_pure']:
                if label_ in resp_visible_differences:
                    resp_visible_differences = resp_visible_differences.replace(label_, '')
            answers.append(resp_visible_differences)
        if len(np.unique(answers)) == len (prompt):
            z_prompts.append([f'{ans}' for ans in answers ])
    return z_prompts

def get_z_prompts_llama(dataset_name, question_prompts):
    tokenizer = LlamaTokenizer.from_pretrained(sys_const.LLAMA_PATH)
    model = LlamaForCausalLM.from_pretrained(sys_const.LLAMA_PATH)
    # question_prompts = text_prompts[dataset_name]['question_llama']
    z_prompts = []
    for prompt in question_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        resp_visible_differences = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip().rstrip()
        resp_visible_differences = resp_visible_differences.split(prompt)[1]
        for label_ in text_prompts[dataset_name]['labels_pure']:
            if label_ in resp_visible_differences:
                resp_visible_differences = resp_visible_differences.replace(label_, '')
        ans = resp_visible_differences.split('. ')
        pattern = r'[0-9]'
        ans = [re.sub(pattern, '', a_.strip().rstrip()).lower() for a_ in ans if len(a_) > 1]
        for i, a_ in enumerate(ans):
            if text_prompts[dataset_name]['forbidden_words'] != None:
                for word in text_prompts[dataset_name]['forbidden_words']:
                    a_ = a_.replace(word, '').strip().rstrip().lower()
                    ans[i] = a_
        ans = [a_ for a_ in ans if len(a_) > 1]
        z_prompts.append(ans)
    z_len = [len(z) for z in z_prompts]
    min_z = np.amin(np.array(z_len))
    for i, z in enumerate(z_prompts):
        if len(z) > min_z:
            z_prompts[i] = np.random.choice(z, min_z, replace=False).tolist()
    z_prompts_final = []
    for i in range(len(z_prompts[0])):
        z_prompts_final.append([z[i] for z in z_prompts])
    if dataset_name == 'waterbirds':
        template = text_prompts[dataset_name]['prompt_template']
        for i, group in enumerate(z_prompts_final):
            for j, item in enumerate(group):
                z = z_prompts_final[i][j].replace('they have', template).replace('they are', template)
                z_prompts_final[i][j] = z
    print('FINAL PROMPTS')
    print(z_prompts_final)
    return z_prompts_final
    

def get_z_prompts_openLM(dataset_name, model_name, question):
    assert model_name in SUPPORTED_MODEL
    if model_name == const.T5_NAME:
        return get_z_prompts_t5(dataset_name, question)
    elif model_name == const.GPT2_NAME:
        return get_z_prompts_gpt2(dataset_name, question)
    elif model_name == const.LLAMA_NAME:
        return get_z_prompts_llama(dataset_name, question)
    
def get_z_prompts_t5(dataset_name, question_prompts):
    model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large')
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
    # question_prompts = text_prompts[dataset_name]['question_openLM']
    prompt_template = text_prompts[dataset_name]['prompt_template'].strip().rstrip()
    z_prompts = []
    for prompt in question_prompts:
        answers = []
        for q in prompt:
            inputs = tokenizer(q, return_tensors="pt")
            outputs = model.generate(**inputs)
            resp_visible_differences = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip().rstrip()
            for substr in resp_visible_differences.split(' '):
                if substr in prompt_template.split(' ')[1:]:
                    resp_visible_differences = resp_visible_differences.replace(substr, '').replace('  ', ' ').lower()
            print(q, resp_visible_differences)
            answers.append(resp_visible_differences)
        for i, a_ in enumerate(answers):
            if text_prompts[dataset_name]['forbidden_words'] != None:
                for word in text_prompts[dataset_name]['forbidden_words']:
                    for w_ in word.split(' '):
                        a_ = a_.replace(w_, '').strip().rstrip().lower()
                    answers[i] = a_
        z_prompts.append([f'{ans}' for ans in answers])
    print('FINAL PROMPTS')
    print(z_prompts)
    return z_prompts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run chatGPT reprompting')
    parser.add_argument('-dataset', '--dataset_name', type=str, required=True)
    args = parser.parse_args()

    model_name = 'llama'
    dataset_name = args.dataset_name
    
    get_z_prompts_openLM(dataset_name, model_name)