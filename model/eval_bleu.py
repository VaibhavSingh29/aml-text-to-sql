from nltk.translate.bleu_score import corpus_bleu
import json

with open('../experiments/t5conditional_decoder_start/predictions_eval_None.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

ref_sentences = []
can_sentences = []

for item in data:
    label = item['label']
    prediction = item['prediction']
    label = ' '.join(label.replace("(", " ( ").replace(")", " ) ").replace("=", ' = ').split())
    prediction = ' '.join(prediction.replace("(", " ( ").replace(")", " ) ").replace("=", ' = ').split())
    ref_sentences.append([label])
    can_sentences.append([prediction])

# ref_sentence = "(((Value=*)count)(Table=song)Project)"
# can_sentence = "(((Value=*)count)(Table=singer)Project)"
# ref_sentence = ' '.join(ref_sentence.replace("(", " ( ").replace(")", " ) ").replace("=", ' = ').split())
# can_sentence = ' '.join(can_sentence.replace("(", " ( ").replace(")", " ) ").replace("=", ' = ').split())
# print(ref_sentences, can_sentences)
# Convert the sentences to lists of words
# ref_sentence = ref_sentence.split()
# can_sentence = can_sentence.split()

# Define the weights for the n-grams
weights = [0.25, 0.25, 0.25, 0.25]

# Calculate the BLEU score
bleu_score = corpus_bleu(ref_sentences, can_sentences, weights=weights)

print("BLEU score:", bleu_score)