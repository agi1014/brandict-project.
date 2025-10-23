import praw
import csv
import os

# --- IMPORTANT: Please use your NEW credentials here ---
# It is highly recommended to use new credentials as the old ones were exposed.
reddit = praw.Reddit(
    client_id="9N_eX3g7zsGX5WZW2oRZKA",
    client_secret="-csSN7l993XGGuHDh2P5NdgLGSkf9w",
    user_agent="Brandict Country Data Collector v2.0"
)

# --- Configuration ---
SEARCH_TERM = "nike"
LIMIT_PER_SUBREDDIT = 75  # Increased limit to get more data
OUTPUT_FILE = 'data/raw_text_data.csv'

# A curated list of country-specific subreddits to get global data
# Maps a country name to a list of relevant subreddits
TARGET_COUNTRIES = {
    "USA": ["AskAnAmerican", "sports", "sneakers"],
    "UK": ["unitedkingdom", "AskUK", "CasualUK"],
    "Canada": ["canada", "askTO", "vancouver"],
    "Australia": ["australia", "sydney", "melbourne"],
    "Germany": ["de", "germany"],
    "France": ["france"],
    "India": ["india", "indiasocial", "mumbai"],
    "Brazil": ["brasil", "futebol"],
    "Japan": ["japanlife"],
    "SouthAfrica": ["southafrica"],
}

print(f"--- Starting Specialized Country-Level Data Collection for '{SEARCH_TERM}' ---")

# Open the file ONCE in write mode ('w') before the loop starts
try:
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write the new header row with the 'country' column
        writer.writerow(['brand', 'country', 'text'])

        # Loop through each target country and its associated subreddits
        for country, subreddits in TARGET_COUNTRIES.items():
            print(f"\n-- Fetching data for country: {country} --")
            for subreddit_name in subreddits:
                print(f"  -> Searching in r/{subreddit_name}...")
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    # Search for the brand name within this specific subreddit
                    for submission in subreddit.search(SEARCH_TERM, limit=LIMIT_PER_SUBREDDIT):
                        # Combine title and body text for more context
                        full_text = submission.title + " " + submission.selftext
                        # Write the brand, the country name, and the text
                        writer.writerow([SEARCH_TERM, country, full_text])
                except Exception as e:
                    print(f"    Could not access r/{subreddit_name}. Error: {e}")

    print(f"\n--- Data collection complete! ---")
    print(f"Results saved to '{OUTPUT_FILE}' with country-level data.")

except Exception as e:
    print(f"An error occurred while trying to write to the file: {e}")
