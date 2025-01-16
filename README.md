# Sentiment Analysis of Tweets Using Python and VADER

This project demonstrates the application of **Python** and advanced **text analysis** techniques to perform **sentiment analysis** on a curated dataset of tweets. The analysis leverages the **VADER (Valence Aware Dictionary and Sentiment Reasoner)** sentiment analysis tool to classify tweets as positive, neutral, or negative. The project combines **JSON** for data storage and **CSV** for potential output extensions, showcasing an end-to-end sentiment analysis pipeline.

## Features

- **Manual Annotation**: A dataset of 25 tweets was manually annotated with sentiment labels (`positive`, `neutral`, `negative`) to establish ground truth for evaluation.
- **Sentiment Analysis with VADER**: Automated sentiment classification was conducted using the VADER model, a tool specifically designed for social media sentiment analysis.
- **Evaluation Pipeline**: Compared VADER's sentiment predictions against manually annotated labels to assess its accuracy and reliability.
- **Data Management**: Utilized **JSON** for structured data storage, making the dataset portable and easily reusable in future projects.

## Why This Project Matters

This project bridges the gap between theoretical knowledge and practical implementation, showcasing essential skills in:

- **Natural Language Processing (NLP)**: Extracting and processing text data.
- **Sentiment Analysis**: Applying state-of-the-art tools like VADER for real-world data classification.
- **Data Annotation and Evaluation**: Creating labeled datasets and evaluating model performance.
- **Data Handling**: Using JSON for structured data storage and preparation for broader applications.

These skills are directly transferable to applications not only in archaeological texts but also in other fields such as social media analytics, customer sentiment evaluation, market research, and more.

## Methodology

1. **Data Collection**: 
   - Curated a dataset of 25 diverse tweets to ensure coverage of positive, neutral, and negative sentiments.
   - Stored tweets in a JSON file with the following fields:
     - `text_of_tweet`: The content of the tweet.
     - `sentiment_label`: Manual annotation of sentiment (`positive`, `neutral`, `negative`).
     - `tweet_url`: Link to the original tweet for reference.

2. **Sentiment Analysis**:
   - Preprocessed tweets using **SpaCy** to tokenize text and prepare input for VADER.
   - Ran the **VADER SentimentIntensityAnalyzer** to classify the sentiment of each tweet.

3. **Evaluation**:
   - Compared VADER's predictions against manually annotated labels.
   - Printed and analyzed the results to identify trends and discrepancies.

## Key Technologies Used

- **Python**: Core programming language for all components.
- **VADER Sentiment Analysis**: Robust, pre-built tool for sentiment classification.
- **SpaCy**: Advanced NLP library used for tokenization and text preprocessing.
- **JSON**: Data format for storing and managing the dataset.
- **CSV**: Potential output format for analysis results.

## Results

The project yielded insights into VADER's performance on text data, highlighting its strengths in handling social media content and potential limitations. This comparative evaluation underscores the importance of manual annotation in validating automated tools.

## Conclusion

This project is a testament to the practical application of **sentiment analysis**. By combining **Python**, **JSON**, and **VADER**, it showcases essential skills for text analysis and serves as a foundation for further exploration in **Natural Language Processing (NLP)**.
