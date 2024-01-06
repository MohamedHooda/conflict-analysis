import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
import os




def get_tweets_by_username(username, leaning):
    array = []
    url = f'https://nitter.rawbit.ninja/{username}'

    end_date = datetime(2023, 11, 1)

    while True:
        driver = webdriver.Chrome()
        driver.get(url)

        elements = driver.find_elements(By.CLASS_NAME, 'timeline-item')


        
        for div in elements:
            try:
                date_html = div.find_element(By.CLASS_NAME, 'tweet-date').get_attribute('innerHTML')
                date_soup = BeautifulSoup(date_html, 'html.parser')
                date_str = date_soup.a['title']

                # Parse the date string into a datetime object
                tweet_date = datetime.strptime(date_str, "%b %d, %Y · %I:%M %p UTC")

                # Break the loop if the tweet is before the end date
                if tweet_date < end_date:
                    driver.close()
                    print("Reached the end date.")
                    df = pd.DataFrame(array)
                    df.to_csv(f'{username}.csv')
                    return

                # Process the tweet
                tweet_stats = div.find_element(By.CLASS_NAME, 'tweet-stats').get_attribute('innerHTML')
                soup = BeautifulSoup(tweet_stats, 'html.parser')
                numbers = [stat.get_text(strip=True).replace(",", "") for stat in soup.find_all(class_="tweet-stat")]

                tweet_id = div.find_element(By.CLASS_NAME, 'tweet-link').get_attribute('href')

                print(tweet_id)

                parsed_url = urlparse(tweet_id)
                path = parsed_url.path

                id = path.split('/')[-1]


                data = {
                    'tweet_id': id,
                    'tweets': div.find_element(By.CLASS_NAME, 'tweet-content').get_attribute('innerHTML'),
                    'dates': date_str,
                    'username': div.find_element(By.CLASS_NAME, 'username').get_attribute('innerHTML'),
                    'url': url,
                    'replies': numbers[0],
                    'retweets': numbers[1],
                    'quote_retweets': numbers[2],
                    'likes': numbers[3],
                    'leaning': leaning
                }

                array.append(data)
            except:
                continue

        # Attempt to click the "show more" button to load more tweets
        try:
            buttons = driver.find_elements(By.CLASS_NAME, "show-more")
            button = buttons[-1]
            button.click()
            resulting_link = driver.current_url
            url = resulting_link
        except:
            print("No more tweets to load or unable to load more.")
            break

        driver.close()

    # Save the data to a CSV file once the loop ends
    df = pd.DataFrame(array)
    print()
    df.to_csv(f'{leaning}/{username}.csv')



df = pd.read_csv('clusters.csv')

for index, row in df.iterrows():
    get_tweets_by_username(row[0], row[1])

# get_tweets_by_username('Israel', 'israel')
