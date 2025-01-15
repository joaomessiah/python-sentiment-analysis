import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def run_vader(nlp, textual_unit, vader_model, lemmatize=False, parts_of_speech_to_consider=set(), verbose=0):
    """
    Run VADER sentiment analysis on a text using SpaCy preprocessing.
    
    :param nlp: SpaCy NLP object
    :param str textual_unit: Text to analyze (e.g., a sentence or multiple sentences)
    :param vader_model: Initialized VADER sentiment analyzer
    :param bool lemmatize: If True, provide lemmas to VADER instead of tokens
    :param set parts_of_speech_to_consider: Set of POS tags to consider (empty = all POS tags)
    :param int verbose: Verbosity level (0 = no output, 1+ = debug info)

    :return: VADER output scores as a dictionary
    :rtype: dict
    """
    doc = nlp(textual_unit)
    input_to_vader = []

    for sent in doc.sents:
        for token in sent:
            
            if verbose >= 2:
                print(f"Token: {token.text}, POS: {token.pos_}")

            # Use lemma or text based on the 'lemmatize' parameter
            word = token.lemma_ if lemmatize and token.lemma_ != "-PRON-" else token.text

            # Add word based on POS filtering
            if not parts_of_speech_to_consider or token.pos_ in parts_of_speech_to_consider:
                input_to_vader.append(word)

    text_for_vader = " ".join(input_to_vader)
    scores = vader_model.polarity_scores(text_for_vader)

    if verbose >= 1:
        print(f"Input Sentence: {textual_unit}")
        print(f"Input to VADER: {text_for_vader}")
        print(f"VADER Output: {scores}")

    return scores


def vader_output_to_label(vader_output):
    """
    Map VADER output to sentiment label to one of the following values:
    a) positive float -> 'positive'
    b) 0.0 -> 'neutral'
    c) negative float -> 'negative'

    :param dict vader_output: Output dictionary from VADER
    :return: Sentiment label ('negative', 'neutral', 'positive')
    :rtype: str
    """
    compound = vader_output["compound"]
    if compound < 0:
        return "negative"
    elif compound == 0.0:
        return "neutral"
    else:
        return "positive"
