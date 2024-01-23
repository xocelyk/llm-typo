def few_shot_qa(question, serialized_choices):
            return [{'role': 'system', 'content': 'You are a helpful assistant. Your task is to answer multiple-choice commonsense questions.'},
            
            {'role': 'user', 'content': 'Q: Where does my body go after I am no longer living?\nA) zombie B) bodycam C) coffin D) graveyard E) funeral'},
            {'role': 'assistant', 'content': 'D) graveyard'},

            {'role': 'user', 'content': 'Q: What might happen to a person with a tumor?\nA) travel B) die of cancer C) cross street D) meet friends E) say words'},
            {'role': 'assistant', 'content': 'B) die of cancer'},

            {'role': 'user', 'content': 'Q: If he did not pass course this semester he would have to what?\nA) learn B) fail C) get certificate D) go back to E) feel proud'},
            {'role': 'assistant', 'content': 'D) go back to'},

            {'role': 'user', 'content': "Q: The knob of the kitchen drawer wouldn't stay tightened, so he went to buy a new screw where?\nA) control panel B) television C) opening mailbox D) supermarket E) hardware store"},
            {'role': 'assistant', 'content': 'E) hardware store'},

            {'role': 'user', 'content': "Q: The small locally owned beauty salon had it's grand opening, people hoped it would boost the economy in the surrounding what?\nA) clerk B) barber shop C) neighborhood D) city E) strip mall"},
            {'role': 'assistant', 'content': 'C) neighborhood'},

            {'role': 'user', 'content': "Q: {}\n{}".format(question, serialized_choices)}]

def few_shot_correct_spelling(question):
        return [{'role': 'system', 'content': 'You are a helpful assistant. Your task is to correct spelling mistakes.'},
                
                {'role': 'user', 'content': "The following question may or may not contain spelling mistakes. Please respond with the corrected question.\Question: Where does my boyd go after I am no longer living?"},
                {'role': 'assistant', 'content': "Where does my body go after I am no longer living?"},

                {'role': 'user', 'content': "The following question may or may not contain spelling mistakes. Please respond with the corrected question.\Question: What might happen to a peesbn with a tumor?"},
                {'role': 'assistant', 'content': "What might happen to a person with a tumor?"},

                {'role': 'user', 'content': "The following question may or may not contain spelling mistakes. Please respond with the corrected question.\Question: If he did not pss oxes this semester he would have to what?"},
                {'role': 'assistant', 'content': "If he did not pass course this semester he would have to what?"},

                {'role': 'user', 'content': "The following question may or may not contain spelling mistakes. Please respond with the corrected question.\Question: The knlb of the kitchen drawer wouldn't stay tightened, so he went to buy a new screw where?"},
                {'role': 'assistant', 'content': "The knob of the kitchen drawer wouldn't stay tightened, so he went to buy a new screw where?"},

                {'role': 'user', 'content': "The following question may or may not contain spelling mistakes. Please respond with the corrected question.\Question: The small locally owned beauty sbllol had it's grand opening, people hoped it would boost the economy in the surrounding what?"},
                {'role': 'assistant', 'content': "The small locally owned beauty salon had it's grand opening, people hoped it would boost the economy in the surrounding what?"},

                {'role': 'user', 'content': "The following question may or may not contain spelling mistakes. Please respond with the corrected question.\Question: {}".format(question)},
            ]