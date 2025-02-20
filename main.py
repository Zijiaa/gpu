import os
import pandas as pd
import requests
from transformers import pipeline

# 🔹 Function to read API URLs from a text file
def load_api_urls(file_path="dataset_urls.txt"):
    """Reads dataset URLs from a text file (one URL per line)."""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: {file_path} not found. Please upload the file.")
        return []
    
    with open(file_path, "r") as file:
        urls = [line.strip() for line in file.readlines() if line.strip()]
    return urls

# 🔹 Function to fetch data from Apify API
def fetch_data_from_api(url):
    """Fetches data from a given Apify dataset API URL and converts it to a DataFrame."""
    print(f"Fetching dataset from API: {url}")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data)
    else:
        print(f"⚠️ Error fetching data from {url}. Status Code: {response.status_code}")
        return pd.DataFrame()

# 🔹 Function to classify sentiment using FinBERT
def get_sentiment_score(text, sentiment_pipeline):
    """Applies FinBERT sentiment analysis and returns a numerical score."""
    try:
        result = sentiment_pipeline(str(text))
        label = result[0]["label"]
        # Convert FinBERT labels to numerical sentiment scores
        if label == "positive":
            return 1.0
        elif label == "negative":
            return -1.0
        else:
            return 0.0
    except Exception as e:
        print(f"Error processing text: {text[:30]}... | {str(e)}")
        return None

# 🔹 Main function to run sentiment analysis
def main():
    # ✅ Load dataset URLs from text file
    dataset_urls = load_api_urls()

    if not dataset_urls:
        print("⚠️ No dataset URLs found. Exiting.")
        return

    all_results = []  # Store all processed datasets

    # ✅ Initialize FinBERT sentiment pipeline with GPU support
    sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert", device=0)

    # ✅ Process each dataset from API
    for url in dataset_urls:
        df = fetch_data_from_api(url)

        # ✅ Check if data is valid
        if df.empty or "fullText" not in df.columns:
            print(f"⚠️ Skipping dataset {url} due to missing or empty 'fullText' column.")
            continue

        # ✅ Compute sentiment scores
        df["sentiment_score"] = df["fullText"].apply(lambda txt: get_sentiment_score(txt, sentiment_pipeline))

        all_results.append(df)

    # ✅ Combine all processed datasets
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        output_file = "sentiment_results_combined.csv"
        final_df.to_csv(output_file, index=False)
        print(f"✅ Sentiment analysis completed! Results saved to '{output_file}' ({len(final_df)} rows).")
    else:
        print("⚠️ No valid data processed. No output file generated.")

# ✅ Run the main function
if __name__ == "__main__":
    main()
