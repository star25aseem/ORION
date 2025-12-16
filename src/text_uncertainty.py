from difflib import SequenceMatcher

def text_uncertainty(responses):
    similarities = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            s = SequenceMatcher(None, responses[i], responses[j]).ratio()
            similarities.append(s)
    return 1 - sum(similarities) / len(similarities)

