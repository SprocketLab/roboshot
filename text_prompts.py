import const
from dataloader import MultiEnvDataset

breeds17_labels = MultiEnvDataset().dataset_dict[const.BREEDS17_NAME]().get_labels()
breeds26_labels = MultiEnvDataset().dataset_dict[const.BREEDS26_NAME]().get_labels()

def get_breeds_question(labels):
    prompt = 'List spurious stereotype of: '
    for i, label_ in enumerate(labels):
        label_ = label_.lower()
        if ',' in label_:
            label_ = label_.split(',')[0]
        prompt += label_
        if i < len(labels)-1:
            prompt += ', '
        else:
            prompt += ', '
    # prompt = 'List the confusing similarities between dog, cat, wolf, and fox. Then list the confusing similarities between ape and monkey. Then list the confusing similarities between grouse and parrot.  Then list the confusing similarities between salamander, lizard, and snake. Then list the confusing similarities between spider and beetle. '
    # prompt = 'List the special physical attributes of dog, cat, wolf, and fox. Then list the special physical attributes of ape and monkey. Then list the special physical attributes of grouse and parrot.  Then list the special physical attributes of salamander, lizard, and snake. Then list the special physical attributes of spider and beetle. '
    prompt+= 'Give one unique answer for each item.'
    return prompt

def get_breeds26_question(labels):
    prompt = 'List the confusing similarities of '
    for i, label_ in enumerate(labels):
        label_ = label_.lower()
        if ',' in label_:
            label_ = label_.split(',')[0]
        prompt += label_
        if i < len(labels)-1:
            prompt += ', '
        else:
            prompt += '. '
    prompt+= 'Give two keywords for each item.'
    return prompt
def get_breeds_forbidden_words(labels):
    forbidden_words = []
    for i, label_ in enumerate(labels):
        label_split = label_.lower().split(', ')
        forbidden_words.extend(label_split)
    return forbidden_words
    
text_prompts = {
    const.WATERBIRDS_NAME: {
        'question_openLM': [
                            ['Waterbirds typically ', 'Landbirds typically '],
                            ['waterbirds usually ', 'landbirds usually '],

                            ['A characteristic of waterbird: ', 'A characteristic of landbird: '],
                            ['Waterbirds are ', 'Landbirds are '],
                            ['A waterbird is ', 'A landbird is '],
                            ['Characteristics of waterbirds: ', 'Characteristics of landbirds: '],
                            ],
        'question_llama': ['List the characteristics of waterbirds: ', 'List the characteristics of landbirds: '],
        'question_reject': 'List the biased/spurious differences between waterbirds and landbirds. Give short keyword for each answer. Answer in the following format: <Difference>: <waterbird characteristic> ; <landbird characteristic>',
        'question_accept': 'List the true visual differences between waterbirds and landbirds. Give short keyword for each answer. Answer in the following format: <Difference>: <waterbird characteristic> ; <landbird characteristic>',
        'object': 'bird',
        'target': 'waterbird',
        'prompt_template': '',
        'labels_pure': ['landbirds', 'waterbirds'],
        'labels': ['an image of landbird', 'an image of waterbird'],
        'forbidden_key': None,
        'forbidden_words': None
    },
    const.CELEBA_NAME: {
        'question_openLM': [
                            ['A visual characteristic of blonde hair: ', 'A visual characteristic of dark hair: '],
                            
                            ['A visual characteristic of blonde person: ', 'A visual characteristic of dair haired person: '],
                            ['A person with blonde hair is generally ', 'A person with dark hair is generally '],
                            ['A person with blonde hair is typically ', 'A person with dark hair is typically '],
                            ['Typical characteristic of people with blonde hair: ', 'Typical characteristic of people with dark hair: '],
                            ['Stereotype of people with blonde hair: ', 'Steretype of people with dark hair: '],
                            ],
        'question_llama': ['List visual characteristics of a blonde haired person: ', 'List visual characteristics of dark haired person: '],
        'question': 'List the true visual differences between blonde haired person and dark haired person. Answer in the following format: <Difference>: <Blonde> ; <Dark>',
        'object': 'hair',
        'target': 'hair color',
        'prompt_template': 'a person with',
        'labels_pure': ['non-blond', 'blond'],
        'labels': ['a person with dark hair', 'a person with blonde hair'],
        'forbidden_key': 'hair color',
        'forbidden_words': ['blonde hair', 'dark hair']
    },
    const.PACS_NAME: {
        'question_openLM': [
                            ['A visual characteristic a dog: ', 'A visual characteristic of an elephant: ','A visual characteristic a giraffe: ','A visual characteristic a guitar: ','A visual characteristic a horse: ','A visual characteristic a house: ','A visual characteristic a person: ',],
                        
                            ['A dog is typically ', 'An elephant is typically ', 'A giraffe is typically ', 'A guitar is typically ', 'A horse is typically ', 'A house is typically ', 'A person is typically '],
                            ['Typical characteristic of a dog: ', 'Typical characteristic of an elephant: ', 'Typical characteristic of a giraffe: ',  'Typical characteristic of a guitar: ',  'Typical characteristic of a horse: ', 'Typical characteristic of a house: ', 'Typical characteristic of a person: ',],
                            ],
        'question_llama': [['List visual characteristics of a dog: ', 
                           'List visual characteristics of an elephant: ',
                           'List visual characteristics of a giraffe: ', 
                           'List visual characteristics of a guitar: ',
                           'List visual characteristics of a horse: ',
                           'List visual characteristics of a house: ',
                           'List visual characteristics of a person: ',],
                           ['List spurious/biased characteristics of a dog: ', 
                           'List spurious/biased characteristics of an elephant: ',
                           'List spurious/biased characteristics of a giraffe: ', 
                           'List spurious/biased characteristics of a guitar: ',
                           'List spurious/biased characteristics of a horse: ',
                           'List spurious/biased characteristics of a house: ',
                           'List spurious/biased characteristics of a person: ',],
                           ],
        # 'question': 'List the difference in visual characteristics of dog, elephant, giraffe and a person. Then, list the similar visual characteristics of house and guitar. Give two keywords for each item. Separate each with a newline',
        'question': 'List the true defining characteristics of dog, elephant, giraffe, person, house and guitar. Give one answer for each. Separate each with a newline',
        'labels_pure': [],
        # ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'],
        'labels': ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'],
        'forbidden_key': None,
        'prompt_template': '',
        'forbidden_key': None,
        'forbidden_words': None,
        # ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'],
    },
    const.VLCS_NAME: {
        'question_openLM': [
                            ['A visual characteristic a bird: ', 'A visual characteristic of a car: ','A visual characteristic a chair: ','A visual characteristic a dog: ','A visual characteristic a person: '],
                        
                            ['A bird is typically ', 'An car is typically ', 'A chair is typically ', 'A dog is typically ', 'A person is typically '],
                            ['Typical characteristic of a bird: ', 'Typical characteristic of an car: ', 'Typical characteristic of a chair: ',  'Typical characteristic of a dog: ',  'Typical characteristic of a person: ',],
                            ],
        'question_llama': [['List visual characteristics of a bird: ', 
                           'List visual characteristics of an car: ',
                           'List visual characteristics of a chair: ', 
                           'List visual characteristics of a dog: ',
                           'List visual characteristics of a person: '],
                           ['List spurious/biased characteristics of a bird: ', 
                           'List spurious/biased characteristics of an car: ',
                           'List spurious/biased characteristics of a chair: ', 
                           'List spurious/biased characteristics of a dog: ',
                           'List spurious/biased characteristics of a person: ']],
        # 'question': 'List the difference in visual characteristics of dog, elephant, giraffe and a person. Then, list the similar visual characteristics of house and guitar. Give two keywords for each item. Separate each with a newline',
        'question': 'List the wrong characteristics of bird, car, chair, dog, person. Give one answer for each item. Separate each with a newline.',
        'labels_pure': [],
        # ['bird', 'car', 'chair', 'dog', 'person'],
        'labels': ['bird', 'car', 'chair', 'dog', 'person'],
        'forbidden_key': None,
        'prompt_template': '',
        'forbidden_key': None,
        'forbidden_words': None
        # ['bird', 'car', 'chair', 'dog', 'person'],
    },
    const.SD_NURSE_FIREFIGHTER_NAME: {
        'question': 'List the biased differences between firefighter and nurse. Give short keywords for each item. Answer in the following format: <Difference>: <Firefighter characteristic> ; <Nurse characteristic>',
        'labels_pure': ['nurse', 'firefighter'],
        'labels': ['nurse', 'firefighter'],
        'forbidden_key': None,
        'prompt_template': 'a person with',
    },
    const.CXR_NAME: {
        'question': 'List the core differences between chest X-ray of person with pneumothorax and without pneumothorax. Give short keywords for each item. Answer in the following format: <Difference>: <pneumothorax characteristic> ; <no pneumothorax characteristic>',
        'labels_pure': ['non-pneumothorax', 'pneumothorax'],
        'labels': ['non-pneumothorax', 'pneumothorax'],
        'forbidden_key': [],
        'prompt_template': '',
    },
    const.BREEDS17_NAME: {
        'question': get_breeds_question(breeds17_labels),
        'labels_pure': get_breeds_forbidden_words(breeds17_labels),
        'labels': get_breeds_forbidden_words(breeds17_labels),
        'forbidden_key': get_breeds_forbidden_words(breeds17_labels),
        'forbidden_words': get_breeds_forbidden_words(breeds17_labels),
        'prompt_template': '',
    },
    const.BREEDS26_NAME: {
        'question': get_breeds26_question(breeds26_labels),
        'labels_pure': breeds26_labels,
        'labels': breeds26_labels,
        'forbidden_key': get_breeds_forbidden_words(breeds26_labels),
        'prompt_template': '',
    }
}