import random
import editdistance

def insert_random_letter(cur, word, i) -> str:
    x = random.choice('abcdefghijklmnopqrstuvwxyz')
    cur.append(x)
    return cur

def delete_last_letter(cur, word, i) -> str:
    return cur

def repeat_letter(cur, word, i) -> str:
    # repeats current letter
    cur.append(word[i])
    cur.append(word[i])
    return cur

def swap_adjacent_letters(cur, word, i) -> str:
    if i == 0 or len(cur) == 0:
        cur.append(word[i])
        return cur
    nxt = word[i]
    temp = cur[-1]
    cur[-1] = nxt
    cur.append(temp)
    return cur

def add_typo(word: str, p: float = 0.1) -> str:
    typo_probs = [(insert_random_letter, 0.4), (delete_last_letter, 0.3), (repeat_letter, 0.1), (swap_adjacent_letters, 0.2)]
    cur = []
    for i in range(len(word)):
        if random.random() < p:
            cur = random.choices(typo_probs, weights=[x[1] for x in typo_probs])[0][0](cur, word, i)
        else:
            cur.append(word[i])
    return ''.join(cur)


def edit_distance(s1: str, s2: str) -> int:
    return editdistance.eval(s1, s2)
