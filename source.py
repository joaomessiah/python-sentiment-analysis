import json
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils import run_vader, vader_output_to_label


def load_tweets(file_path):
    """Load tweets from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON file.")
        return {}

def analyze_tweets(tweets_data, nlp, vader_model):
    """
    Analyze tweets with VADER and compare them with manual annotations.

    :param tweets_data: Dictionary of tweet data
    :param nlp: SpaCy NLP object
    :param vader_model: VADER Sentiment Intensity Analyzer
    :return: Lists of tweets, VADER output, and manual annotations
    """
    tweets = []
    all_vader_output = []
    manual_annotation = []

    for id_, tweet_info in tweets_data.items():
        the_tweet = tweet_info.get("text_of_tweet", "")
        manual_label = tweet_info.get("sentiment_label", "unknown")

        vader_output = run_vader(nlp, the_tweet, vader_model, lemmatize=False)
        vader_label = vader_output_to_label(vader_output)

        tweets.append(the_tweet)
        all_vader_output.append(vader_label)
        manual_annotation.append(manual_label)

    return tweets, all_vader_output, manual_annotation


if __name__ == "__main__":
    # Initialize models
    nlp = spacy.load("en_core_web_sm")
    vader_model = SentimentIntensityAnalyzer()

    # Load tweets
    tweets_data = load_tweets("tweets.json")
    if not tweets_data:
        exit("No tweets to process. Exiting.")

    # Analyze tweets
    tweets, vader_labels, manual_annotations = analyze_tweets(tweets_data, nlp, vader_model)

    # Output results
    print("VADER Output Labels:", vader_labels)
    print("Manual Annotations:", manual_annotations)
