from utils import *
from transformers import BertTokenizer, BertForMaskedLM
import json
import torch
from models import *
import random
from prompts import *
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
import torch

few_shot_questions = ['Where does my body go after I am no longer living?',
                      'What might happen to a person with a tumor?',
                      'If he did not pass course this semester he would have to what?',
                      "The knob of the kitchen drawer wouldn't stay tightened, so he went to buy a new screw where?",
                      "The small locally owned beauty salon had it's grand opening, people hoped it would boost the economy in the surrounding what?"]

def create_misspelling_few_shot(question, p=0.4):
    '''
    Create a misspelling of a few shot question
    '''
    data = load_data()
    for example in data:
        if example['Question'] == question:
            return create_typos(example['Question'], example['StartIndex'], example['EndIndex'], p=p, num_samples=1)[0]['TypoQuestion']

def get_prompt(question: str, serialized_choices: list[str]) -> str:
    serialized_choices = ' '.join(serialized_choices)
    prompt = few_shot_qa(question, serialized_choices)
    return prompt

def query(messages):
    return chatgpt(messages=messages, model="gpt-3.5-turbo-1106", max_tokens=1000, temperature=0)[0]

def load_data():
    '''
    data = 
    {question: str,
    start_index: int, (index of start of word to change)
    end_index: int, (index of end of word to change)
    answer choices: list of str,
    correct answer: str}
    }
    '''
    train_filename = '../data/commonsenseqa/train_rand_split.jsonl'
    
    data = []
    with open(train_filename, 'r') as f:
        for line in f:
            example = load_example(json.loads(line))
            if example:
                data.append(example)
    return data

def load_example(example: dict):
    # TODO: double check no double instance of concept word
    # TODO: what is the question concept? is it always a single word? in the question? why should we perturb it alone?
    question = example['question']['stem']
    concept = example['question']['question_concept']
    if not check_concept_in_question(question, concept):
        return None
    if question in few_shot_questions:
        return None
    start_index = question.find(concept)
    end_index = start_index + len(concept)
    serialized_choices = []
    correct_label = example['answerKey']
    for choice in example['question']['choices']:
        serialized_choice = choice['label'] + ') ' + choice['text']
        if choice['label'] == correct_label:
            correct_answer = serialized_choice
        serialized_choices.append(serialized_choice)
    return {'Question': question, 'StartIndex': start_index, 'EndIndex': end_index, 'AnswerChoices': serialized_choices, 'CorrectAnswer': correct_answer}

def check_concept_in_question(question: str, concept: str) -> bool:
    if concept not in question:
        return False
    start = question.find(concept)
    end = start + len(concept)
    if start > 0 and question[start - 1] != ' ': # check if concept is only part of a word
        return False
    if end < len(question) and question[end] != ' ': # check if concept is only part of a word
        return False
    return True

def create_typos(question: list, start_index: int, end_index: int, p: float = 0.1, num_samples = 5) -> list:
    word = question[start_index:end_index]
    typos = []
    for _ in range(num_samples):
        typo_word = add_typo(word, p=p)
        typos.append(typo_word)

    res = []
    for typo in typos:
        typo_question = [char for char in question]
        typo_question[start_index:end_index] = typo
        typo_question = "".join(typo_question)
        dist = edit_distance(word, typo)
        res.append({'OriginalQuestion': ''.join(question), 'TypoQuestion': ''.join(typo_question), 'TargetWord': word, 'TypoWord': typo, 'EditDistance': dist, 'TargetStartIndex': start_index, 'TargetEndIndex': start_index + len(typo), 'p': p})
    return res

def get_probability_of_word(question: list, start_index: int, end_index, model):
    '''
    Return the probability of the word at index `index` in `sent` using a BERT model.

    Args:
    sent (list): The sentence as a list of words.
    index (int): The index of the target word.
    model (BertForMaskedLM): The BERT model.

    Returns:
    float: The probability of the word at index `index`.
    '''

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Mask the target word
    masked_question = [char for char in question]
    masked_question[start_index: end_index] = tokenizer.mask_token
    masked_question = "".join(masked_question)

    # Tokenize the input sentence and convert to tensor
    inputs = tokenizer(masked_question, return_tensors="pt")

    # Predict all tokens
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    # Get the token id of the original word
    original_word_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question[start_index: end_index])[0])

    # Find the logits for the masked token
    mask_token_index = torch.where(inputs.input_ids[0] == tokenizer.mask_token_id)[0]
    mask_token_logits = predictions[0, mask_token_index, :]

    # Calculate the probability of the original word
    mask_token_probs = torch.softmax(mask_token_logits, dim=-1)
    word_probability = mask_token_probs[0, original_word_id]

    return word_probability.item()

def parse_answer(ans):
    # maybe todo
    return ans.strip()

def main():
    bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    '''
    data = 
    {question: str,
    start_index: int, (index of start of word to change)
    end_index: int, (index of end of word to change)
    answer choices: list of str,
    correct answer: str}
    }
    '''
    
    data = load_data()
    data = data[:300]

    typo_data = []
    p_range = [0.1, 0.2, 0.4, 0.8]
    num_samples = 1
    for example in data:
        start_index = example['StartIndex']
        end_index = example['EndIndex']
        question = example['Question']
        prob_word = get_probability_of_word(question, start_index, end_index, bert_model)
        for p in [0] + p_range:
            if p == 0:
                typos = create_typos(question, start_index, end_index, p=p, num_samples=1)
            else:
                typos = create_typos(question, start_index, end_index, p=p, num_samples=num_samples)
            for typo in typos:
                typo['ProbWord'] = prob_word
                for k, v in example.items():
                    typo[k] = v
            typo_data.extend(typos)
    
    results = []
    for typo in typo_data:
        prompt = get_prompt(typo['TypoQuestion'], typo['AnswerChoices'])
        cache_prompt = prompt.copy()
        ans = query(prompt)
        ans = parse_answer(ans)
        print('Gold Label: \t\t' + typo['CorrectAnswer'])
        print('Question: \t\t' + typo['TypoQuestion'])
        
        example_data = typo
        example_data['Answer'] = ans
        example_data['CorrectAnswer'] = typo['CorrectAnswer']
        example_data['Correct'] = ans.lower() == typo['CorrectAnswer'].lower()
        valid = ans[1] == ')'
        example_data['Valid'] = valid
        corrected_question = query(few_shot_correct_spelling(typo['TypoQuestion']))
        corrected_question = parse_answer(corrected_question)
        prompt = get_prompt(corrected_question, typo['AnswerChoices'])
        if corrected_question == typo['TypoQuestion']:
            assert prompt == cache_prompt
        corrected_ans = query(prompt)
        corrected_ans = parse_answer(corrected_ans)
        if corrected_question == typo['TypoQuestion']:
            assert corrected_ans == ans, 'Original Prompt: {}\nCorrected Prompt: {}'.format(cache_prompt, prompt)
        example_data['CorrectedQuestion'] = corrected_question
        example_data['CorrectedAnswer'] = parse_answer(corrected_ans)
        example_data['CorrectedCorrect'] = corrected_ans.lower() == typo['CorrectAnswer'].lower()
        print('Corrected Question: \t' + corrected_question)
        print('Answer: \t\t' + ans)
        print('Corrected Answer: \t' + corrected_ans)
        print()
        results.append(example_data)
    
        df = pd.DataFrame(results)
        df.to_csv('results_experiment_3.csv')


if __name__ == "__main__":
    main()
