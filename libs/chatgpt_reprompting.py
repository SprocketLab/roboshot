import sys
sys.path.insert(0, '../../')
from utils.api_key import API_KEY

import openai
openai.api_key = API_KEY

from libs.text_prompts import text_prompts
import utils.const as const

import argparse
from tqdm import tqdm

def request(prompt, max_tokens=200):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        top_p=0.8,
    )
    generated = response['choices'][0]['text'].strip().rstrip()
    return generated

def items_to_list(items_str, dataset_name, verbose):
    items = items_str.split('\n')
    if items_str[0] == '-':
        for i, item in enumerate(items):
           items[i] = f'{str(i+1)}. ' + items[1:]
    if 'vs.' in items_str:
        items_str = items_str.replace('vs.', 'vs')
    items = [i_.split('. ')[-1].lower() for i_ in items]
    items = [i_[:-1] if '.' in i_ else i_ for i_ in items]

    if ':' in items_str:
        split_str = ':'
    elif '-' in items_str:
        split_str = '-'
    
    keys = [i_.split(split_str)[0].lower() for i_ in items if len(i_.split(split_str)[0]) > 1]
    keys = [i_.strip().rstrip() for i_ in keys]
    values = [i_.split(split_str)[1].lower() for i_ in items if len(i_.split(split_str)[0]) > 1]
    values = [i_.strip().rstrip() for i_ in values]
    for i, val in enumerate(values):
        if ',' in val:
            split_str_val = ','
        elif ';' in val:
            split_str_val = ';'
        # elif '/' in val:
        #     split_str_val = '/'
        elif 'vs.' in val:
            split_str_val = 'vs.'
        elif 'vs' in val:
            split_str_val = 'vs'
        elif ';' in val:
            split_str_val = ';'
        elif '-' in val:
            split_str_val = '-'
        elif '(' in val:
            split_str_val = '('
        else:
            split_str_val = ':'
        values[i] = val.split(split_str_val)
        # if len(values) < 2:
        #     i1 = values[i][0]
        #     answers = []
        #     for label_ in text_prompts[dataset_name]['labels']:
        #         prompt = f"Answer with one word. What is the {keys[i]} for {i1}?"
        #         ans = request(prompt, max_tokens=3)
        #         answers.append(ans)
        #     i1, i2 = answers
    for i, vals in enumerate(values):
        values_clean = []
        for v in vals:
            for label_ in text_prompts[dataset_name]['labels_pure']:
                # v = re.sub('[^0-9a-zA-Z]+', ' ', v)
                v = v.replace(label_, '')
                # v = v.split('-')[-1]
                values_clean.append(v.strip().rstrip())
        values[i] = list(set([v for v in values_clean if len(v)>0]))
    kv_dict = {}
    for i, key in enumerate(keys):
        if len(values[i]) ==1:
            continue
        # if values[i][0] == values[i][1]:
        #     continue
        # if dataset_name == 'waterbirds':
        #     verification_prompt = f'Answer with a yes/no. Can we see {key} of {text_prompts[dataset_name]["object"]} in a photograph?'
        #     answer1 = request(verification_prompt, max_tokens=3)
        #     if 'yes' in answer1.lower():
        #         kv_dict[key] = values[i]
        # else:
        kv_dict[key] = list(set(values[i]))
    if len(kv_dict) == 0:
        i1, i2 = text_prompts[dataset_name]['labels']
        for i, key in enumerate(keys):
            # values = []
            prompt = f"Answer with one word. What is the {keys[i]} for {i1}?"
            ans1 = request(prompt, max_tokens=3)
            prompt = f"Answer with one word. What is the {keys[i]} for {i2}?"
            ans2 = request(prompt, max_tokens=3)
            kv_dict[key] = [ans1, ans2]
    # print(kv_dict)
    return kv_dict

def construct_final_prompt(kv_dict, dataset_name):
    prompts = []
    for k,v in kv_dict.items():
        if '[TEMPLATE]' not in text_prompts[dataset_name]["prompt_template"]:
            for i in range(len(v)):
                # if (text_prompts[dataset_name]['labels_pure'] and text_prompts[dataset_name]['forbidden_key']) and (k not in text_prompts[dataset_name]['labels_pure']) and (k not in text_prompts[dataset_name]['forbidden_key']):
                prompt = [f'{text_prompts[dataset_name]["prompt_template"]} {v[i]} {k}' for i in range(len(v))]
                # else:
                #     prompt = [f'{text_prompts[dataset_name]["prompt_template"]} {v[i]}' for i in range(len(v))]
            for i, p in enumerate(prompt):
                p = ''.join(c for c in p if c.isalnum() or c == ' ')
                prompt[i] = p.strip().rstrip().replace('  ', ' ')
            prompts.append(prompt)
        else:
            for i in range(len(v)):
                # if (text_prompts[dataset_name]['labels_pure'] and text_prompts[dataset_name]['forbidden_key']) and (k not in text_prompts[dataset_name]['labels_pure']) and (k not in text_prompts[dataset_name]['forbidden_key']):
                prompt = [text_prompts[dataset_name]["prompt_template"].replace('[TEMPLATE]',f'{v[i]} {k}') for i in range(len(v))]
                # else:
                #     prompt = [text_prompts[dataset_name]["prompt_template"].replace('[TEMPLATE]',f'{v[i]}') for i in range(len(v))]
            for i, p in enumerate(prompt):
                p = ''.join(c for c in p if c.isalnum() or c == ' ')
                prompt[i] = p.strip().rstrip().replace('  ', ' ')
            prompts.append(prompt)
    return prompts

def get_z_prompts(dataset_name, question, verbose=True, n_paraphrases=1, max_tokens=100):
    #step 1
    resp_visible_differences = request(question, max_tokens=max_tokens)
    if verbose:
        print("########## original response ##########")
        print(resp_visible_differences)
        # exit()

    #step 2
    kv_dict = items_to_list(resp_visible_differences, dataset_name, verbose)
    #step 4
    prompts = construct_final_prompt(kv_dict, dataset_name)
    if dataset_name != const.CXR_NAME:
        if verbose:
            print("getting paraphrases...")
        paraphrased_prompts = []
        prompt_template_start = text_prompts[dataset_name]['prompt_template'].split('[TEMPLATE]')[0].lower().split(' ')[0]
        if len(prompts[0]) == 2:
            for i in range(n_paraphrases):
                for j, item in tqdm(enumerate(prompts)):
                    try:
                        p1, p2 = item
                        prompt_1 = f"Give me a short paraphrase for: {p1}. "
                        paraphrase_p1 = request(prompt_1, max_tokens=10).replace('\n', '').replace('.', '').strip().rstrip().lower()
                        print(p1, "|", paraphrase_p1)
                        prompt_1 = f"Give me a short paraphrase for: {p2}. "
                        paraphrase_p2 = request(prompt_1, max_tokens=10).replace('\n', '').replace('.', '').strip().rstrip().lower()
                        print(p2, "|", paraphrase_p2)
                        if paraphrase_p1.startswith(prompt_template_start) and paraphrase_p2.startswith(prompt_template_start):
                            paraphrased_prompts.append([paraphrase_p1, paraphrase_p2])
                    except:
                        continue
            prompts.extend(paraphrased_prompts)
    # prompts = list(set(prompts))
    if verbose:
        print('FINAL PROMPT')
        print(prompts)
    return prompts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run chatGPT reprompting')
    parser.add_argument('-dataset', '--dataset_name', type=str, required=True)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    
    get_z_prompts(dataset_name)