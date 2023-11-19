import nltk
import string
from nltk.corpus import stopwords
import numpy as np

nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # For stopwords
nltk.download('averaged_perceptron_tagger')  # For POS tagging


def extract_posTag_features(text_posTag):
    adjective_tags = ['JJ', 'JJ$', 'JJ+JJ', 'JJR', 'JJR+CS', 'JJS', 'JJT', ]
    noun_tags = ['NN', 'NN$', 'NN+BEZ', 'NN+HVD', 'NN+HVZ', 'NN+IN', 'NN+MD', 'NN+NN', 'NNS', 'NNS$', 'NNS+MD', 'NP', 'NP$', 'NP+BEZ', 'NP+HVZ', 'NP+MD', 'NPS', 'NPS$', 'NR', 'NR$', 'NR+MD', 'NRS']
    pronoun_tags = ['PN', 'PN$', 'PN+BEZ', 'PN+HVD', 'PN+HVZ', 'PN+MD', 'PP$$', 'PPL', 'PPLS', 'PPO', 'PPS', 'PPS+BEZ', 'PPS+HVD', 'PPS+HVZ', 'PPS+MD', 'PPSS', 'PPSS+BEM', 'PPSS+BER', 'PPSS+BEZ', 'PPSS+BEZ*', 'PPSS+HV', 'PPSS+HVD', 'PPSS+MD', 'PPSS+VB']
    article_tags = ['AT']
    conjunction_tags = ['CC', 'CS']
    numeral_tags = ['CD', 'CD$', 'OD']
    preposition_tags = ['IN', 'IN+IN', 'IN+PPO']
    qualifier_tags = ['QL', 'QLP']
    adverb_tags = ['RB', 'RB$', 'RB+BEZ', 'RB+CS', 'RBR', 'RBR+CS', 'RBT', 'RN', 'RP', 'RP+IN']
    foreign_word = 'FW'
    w_classification_tags = ['WDT', 'WDT+BER', 'WDT+BER+PP', 'WDT+BEZ', 'WDT+DO+PPS', 'WDT+DOD', 'WDT+HVZ', 'WP$', 'WPO', 'WPS', 'WPS+BEZ', 'WPS+HVD', 'WPS+HVZ', 'WPS+MD', 'WQL', 'WRB','WRB+BER', 'WRB+BEZ', 'WRB+DO', 'WRB+DOD', 'WRB+DOD*', 'WRB+DOZ', 'WRB+IN', 'WRB+MD']
    modal_tags = ['MD', 'MD*', 'MD+HV', 'MD+PPSS', 'MD+TO']
    a_determiner_tags = ['ABL', 'ABN', 'ABX', 'AP', 'AP$', 'AP+AP']
    determiner_tags = ['DT', 'DT$', 'DT+BEZ', 'DT+MD', 'DTI', 'DTS', 'DTX']
    verb_tags = ['BE', 'BED', 'BED*', 'BEDZ', 'BEDZ*', 'BEG', 'BEM', 'BEM*', 'BEN', 'BER', 'BER*', 'BEZ', 'BEZ*', 'DO', 'DO*', 'DO+PPSS', 'DOD', 'DOD*', 'DOZ', 'DOZ*', 'HV','HV*', 'HV+TO', 'HVD', 'HVD*', 'HVG', 'HVN', 'HVZ', 'HVZ*', 'VB', 'VB+AT', 'VB+IN', 'VB+JJ', 'VB+PPO', 'VB+RP', 'VB+TO', 'VB+VB', 'VBD', 'VBG', 'VBG+TO', 'VBN','VBN+TO', 'VBZ']

    n_posTags = len(text_posTag)

    adjectives = 0
    nouns = 0
    pronouns = 0
    articles = 0
    conjunctions = 0
    numerals = 0
    prepositions = 0
    qualifiers = 0
    adverbs = 0
    foreign_words = 0
    w_classifications = 0
    modals = 0
    a_determiners = 0
    determiners = 0
    verbs = 0
    for tag in text_posTag:
        if tag in adjective_tags:
            adjectives += 1
        elif tag in noun_tags:
            nouns += 1
        elif tag in pronoun_tags:
            pronouns += 1
        elif tag in article_tags:
            articles += 1
        elif tag in conjunction_tags:
            conjunctions += 1
        elif tag in numeral_tags:
            numerals += 1
        elif tag in preposition_tags:
            prepositions += 1
        elif tag in qualifier_tags:
            qualifiers += 1
        elif tag in adverb_tags:
            adverbs += 1
        elif foreign_word in tag:
            foreign_words += 1
        elif tag in w_classification_tags:
            w_classifications += 1
        elif tag in modal_tags:
            modals += 1
        elif tag in a_determiner_tags:
            a_determiners += 1
        elif tag in determiner_tags:
            determiners += 1
        elif tag in verb_tags:
            verbs += 1

    adjectives_ratio = adjectives/n_posTags
    nouns_ratio = nouns/n_posTags
    pronouns_ratio = pronouns/n_posTags
    articles_ratio = articles/n_posTags
    conjunctions_ratio = conjunctions/n_posTags
    numerals_ratio = numerals/n_posTags
    prepositions_ratio = prepositions/n_posTags
    qualifiers_ratio = qualifiers/n_posTags
    adverbs_ratio = adverbs/n_posTags
    foreign_words_ratio = foreign_words/n_posTags
    w_classifications_ratio = w_classifications/n_posTags
    modals_ratio = modals/n_posTags
    a_determiners_ratio = a_determiners/n_posTags
    determiners_ratio = determiners/n_posTags
    verbs_ratio = verbs/n_posTags

    if adjectives_ratio > 1 or nouns_ratio > 1:
        print(f'Len_posTags: {n_posTags} --> Adjectives: {adjectives}, --> Nouns: {nouns}')

    return adjectives_ratio, nouns_ratio, conjunctions_ratio, prepositions_ratio, adverbs_ratio, w_classifications_ratio, modals_ratio, determiners_ratio, verbs_ratio


def extract_stopwords_and_ponctuation_ratio(text):

    stopwords_count = 0
    pontucation_count = 0


    from nltk.corpus import stopwords
    stopwords_list = stopwords.words("english")
    tokens = nltk.word_tokenize(text)
    n_tokens = len(tokens)

    for token in tokens:
        if token in stopwords_list:
            stopwords_count += 1
        elif token in string.punctuation:
            pontucation_count += 1

    stopwords_ratio = stopwords_count/n_tokens
    pontucation_ratio = pontucation_count/n_tokens

    if stopwords_ratio > 1 or pontucation_ratio > 1:
        print(f'Len_tokens: {n_tokens} --> Stopwords: {stopwords_count}, --> Ponctuation: {pontucation_count}')

    return stopwords_ratio, pontucation_ratio


def extract_average_token_quantity_per_sentence(text):
    tokens_quantity = []
    sentences =  nltk.sent_tokenize(text)

    for sentence in sentences:
        sentence_tokens = nltk.word_tokenize(sentence)
        tokens_quantity.append(len(sentence_tokens))

    average_tokens_quantity_per_sentence = sum(tokens_quantity)//len(tokens_quantity)

    return average_tokens_quantity_per_sentence
    
    

def extract_average_token_length_per_text(text):

    tokens_length = []
    tokens = nltk.word_tokenize(text)

    for token in tokens:
        tokens_length.append(len(token))

    average_token_length = sum(tokens_length)//len(tokens_length)

    return average_token_length


def extract_sentence_quantity_per_text(text):
    sentences_quantity =  len(nltk.sent_tokenize(text))

    if sentences_quantity > 300:
        print(f'To many sentences: {text}')

    return sentences_quantity
    


def extract_features(text, text_posTag, label=None, dataset=None, min_word_size=1, max_word_size=43, min_n_sentences_per_text=3, max_n_sentences_per_text=116,  min_n_words_per_sentence=1, max_n_words_per_sentence=943 ):
    
    adjectives_ratio, nouns_ratio, conjunctions_ratio, prepositions_ratio, adverbs_ratio, w_classifications_ratio, modals_ratio, determiners_ratio, verbs_ratio = extract_posTag_features(text_posTag)

    stopwords_ratio, ponctuation_ratio = extract_stopwords_and_ponctuation_ratio(text)

    average_token_quantity_per_sentence = extract_average_token_quantity_per_sentence(text)

    average_token_length_per_text = extract_average_token_length_per_text(text)

    
    return [adjectives_ratio, nouns_ratio, conjunctions_ratio, prepositions_ratio, adverbs_ratio, w_classifications_ratio, modals_ratio, determiners_ratio, verbs_ratio, stopwords_ratio, ponctuation_ratio, average_token_quantity_per_sentence, average_token_length_per_text]

def normalizers(data):
    normalizers = []
    n_features = len(data[0])

    for index in range(n_features):
        features = np.array([[feature[index]] for feature in full_dataset])
        scaler = MinMaxScaler()
        scaler.fit(features)
        normalizers.append(scaler)

    return normalizers

def normalize_features(data, normalizers):
    normalized_features = []
    for index, feature in enumerate(data):
        normalized_features.append(normalizers[index].transform([[data[index]]])[0][0])

    return normalized_features