import pandas as pd
import requests
from bs4 import BeautifulSoup

# Using the "Quotes to Scrape" website
url = 'https://quotes.toscrape.com'
current_url = url + '/'
extracted_data = []
num_page = 1

# Extracting the information
while True:
    print(f"Scraping page {num_page}...")
    response = requests.get(current_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    containers = soup.find_all('div', class_='quote')
    for box in containers:
        phrase = box.find('span', class_='text').text
        author = box.find('small', class_='author').text
        extracted_data.append({'phrase':phrase, 'author':author})
    next_button = soup.find('li', class_='next')
    if next_button:
        relative_link = next_button.find('a')['href']
        url_current = url + relative_link
        num_page += 1
    else:
        print("There are no more pages")
        break 

# Displaying the data
print("------------------------------------")
print(f"A total of {len(extracted_data)} observations were extracted")
df = pd.DataFrame(extracted_data)
print(df.head())