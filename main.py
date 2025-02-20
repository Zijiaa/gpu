# main.py
import pandas as pd
from transformers import pipeline

# Define macroeconomic topics with related keywords
macro_topics = {
    "inflation": [
        "inflation", "flation", "cpi", "consumer price index", "product price index", 
        "ppi", "price stability", "consumer prices", "shrinkflation", "disinflationary", 
        "disinflation", "cost of living", "deflation", "hyperinflation", 
        "inflation expectations", "stagflation", "headline inflation", "core inflation"
    ],
    "unemployment": [
        "unemployment","job","job growth", "labor market", "labor", "wages", 
        "employment", "jobless", "hiring", "underemployment", "workforce", "layoffs"
    ],
    "interest_rates": [
        "interest rate", "fed rates","federal reserve","fedral open market committee",
        "monetary", "monetary policy", "fomc", "rate hike", "rate cut", 
        "quantitative easing", "qe", "quantitative tightening", "yield curve", "ffr",
        "federal funds rate"
    ],
    "economic_growth": [
        "gdp","domestic product","gnp", "gross national product","economic expansion", 
        "recession", "growth", "economic slowdown", "stagnation", "boom and bust cycle", 
        "business cycle", "productivity", "economic indicators"
    ],
    "housing": [
        "mortgage","housing","home construction"
    ],
}

def classify_topic(text):
    """
    Assigns a macroeconomic topic to the given text 
    by matching keywords. Returns 'other' if no match.
    """
    text_lower = str(text).lower()
    for topic, keywords in macro_topics.items():
        # If *any* keyword is in the text, return that topic
        if any(keyword in text_lower for keyword in keywords):
            return topic
    return "other"

def get_sentiment_score(text, sentiment_pipeline):
    """
    Applies FinBERT pipeline on the text and returns a
    numerical sentiment score: +1 for positive, -1 for negative, 0 for neutral.
    """
    result = sentiment_pipeline(str(text))
    label = result[0]["label"]
    # Convert FinBERT labels to numerical sentiment scores
    if label == "positive":
        return 1.0
    elif label == "negative":
        return -1.0
    else:
        return 0.0

def main():
    # 1) Load dataset
    file_path = "sorted_extracted_tweet_data.csv"
    df = pd.read_csv(file_path)

    # 2) Validate columns
    required_cols = ["fullText", "createdAt"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")

    # 3) Classify topics
    df["topic"] = df["fullText"].apply(classify_topic)

    # 4) Initialize FinBERT pipeline
    #    Use device=0 to enable GPU if your environment has one available.
    sentiment_pipeline = pipeline(
        "text-classification", 
        model="ProsusAI/finbert", 
        device=0  # <== Set to 0 for GPU, -1 for CPU
    )

    # 5) Compute sentiment scores
    df["sentiment_score"] = df["fullText"].apply(
        lambda txt: get_sentiment_score(txt, sentiment_pipeline)
    )

    # 6) Convert 'createdAt' to date
    df["date"] = pd.to_datetime(df["createdAt"], errors='coerce').dt.date
    df.dropna(subset=["date"], inplace=True)  # remove invalid dates

    # 7) Group by topic and date, average sentiment
    sentiment_summary = (
        df.groupby(["topic", "date"])["sentiment_score"]
        .mean()
        .reset_index()
    )

    # 8) Save results
    output_file = "aggregated_sentiment_by_topic_finbert.csv"
    sentiment_summary.to_csv(output_file, index=False)

    print("Sentiment analysis and topic classification completed.")
    print(f"Results saved to '{output_file}'. Length: {len(sentiment_summary)} rows")

if __name__ == "__main__":
    main()
