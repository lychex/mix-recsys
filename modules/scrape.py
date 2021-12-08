import re

import lxml
import pandas as pd
import requests
from bs4 import BeautifulSoup


def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    # Remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)
    text = text.strip()
    return text


def clean_digits(text):
    return int(text.split()[1].replace(',', ''))


def get_data(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup


def get_div_list(soup):
    num_list = ['One', 'Two', 'Three', 'Four',
                'Five', 'Six', 'Seven', 'Eight', 'Nine']
    div_list = []
    for num in num_list:
        link = soup.find('li', {'class': 'subNav'+num}).find('a')['href']
        div_list.append(link)
    return div_list


def get_all_topics(div_list):
    all_topic_titles = []
    all_topic_links = []

    for div_link in div_list:
        topic_titles = []
        topic_links = []

        sp = get_data(div_link)
        results = sp.find_all('li', {'class': 'cat-item'})
        for i, item in enumerate(results):
            topic_titles.append(clean_text(item.find('a').contents[0]))
            topic_links.append(item.find('a')['href'])
        all_topic_titles.extend(topic_titles[1:])
        all_topic_links.extend(topic_links[1:])
    return list(zip(all_topic_titles, all_topic_links))


def get_topic_detail(topic_link):

    big_list = []

    for x in range(15):
        try:
            topic_link_page = topic_link + f'/page/{x+1}'

            s = get_data(topic_link_page)

            small_list = []
            results = s.find_all('div', {'class': 'your-stories__text'})
            for article in results:
                item = article.find('a').contents[0]
                item = clean_text(item).strip()
                small_list.append(item)
            big_list.extend(small_list)
        except:
            pass

    return big_list


# soup = get_data(url)
# div_list = get_div_list(soup)
# div_list[5] = div_list[5] + '/benefits'
# print(div_list)


# topic_catelog = dict()
# for topic_title, topic_link in all_topics:
#     topic_catelog[topic_title] = get_topic_detail(topic_link)
# print(topic_catelog)

def get_topic_info(topic_link):

    big_list_title = []
    big_list_view = []

    for x in range(15):
        try:
            topic_link_page = topic_link + f'/page/{x+1}'

            sp = get_data(topic_link_page)

            title_results = sp.find_all('div', {'class': 'your-stories__text'})
            small_list_title = []
            for title_item in title_results:
                title = title_item.find('a').contents[0]
                title = clean_text(title)
                small_list_title.append(title)

            link_results = sp.find_all(
                'div', {'class': 'your-stories__content'})
            small_list_view = []
            for link_item in link_results:
                sub_link = link_item.find('a')['href']

                s = get_data(sub_link)
                view = s.find("span", title=re.compile("Page Views")).text
                view = clean_digits(view)
                small_list_view.append(view)

            big_list_title.extend(small_list_title)
            big_list_view.extend(small_list_view)
        except:
            pass

    return big_list_title, big_list_view

#############################################################################
#                          Get Support Tab
#############################################################################


def get_categories(url):
    soup = get_data(url)
    results = soup.find_all('h3', {'class': 'CategoryNameHeading'})
    categories = []
    cat_links = []
    for item in results:
        categories.append(clean_text(item.find('a').contents[0]))
        cat_links.append(item.find('a')['href'])
    return categories, cat_links


#############################################################################
#                          News and Research Tab
#############################################################################


def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    # Remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)
    # Remove whitespace
    text = text.strip()
    return text


def get_data(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup


def get_sub_list(soup):
    num_list = ['One', 'Two', 'Three', 'Four',
                'Five', 'Six', 'Seven']
    div_list = []
    for num in num_list:
        link = soup.find('li', {'class': 'subNav'+num}).find('a')['href']
        div_list.append(link)
    return div_list


news_url = 'https://www.themix.org.uk/news-and-research/news'


def get_name_and_link(news_url):
    soup = get_data(news_url)
    cat_list = ['menu-item-17440', 'menu-item-19129', 'menu-item-19297',
                'menu-item-18049', 'menu-item-17767', 'menu-item-17439']

    names = []
    links = []
    for cat in cat_list:
        result = soup.find('li', {'class': cat})
        names.append(clean_text(result.find('a').text))
        links.append(result.find('a')['href'])
    return names, links


news_main_link = 'https://www.themix.org.uk/news-and-research/news/'


def get_news_info(news_main_link):

    big_list_name = []
    big_list_view = []
    for x in range(15):
        try:
            sub_link = news_main_link + f'/page/{x+1}'
            soup = get_data(sub_link)

            name_list = []
            view_list = []

            results = soup.find_all('div', {'class': 'your-stories__content'})
            for sec in results:
                fin_link = sec.find('a')['href']
                s = get_data(fin_link)
                title = clean_text(s.find('h1', {'class': 'label-news'}).text)
                view = s.find("span", title=re.compile("Page Views")).text

                name_list.append(title)
                view_list.append(view)

            big_list_name.extend(name_list)
            big_list_view.extend(view_list)

        except:
            pass
    return big_list_name, big_list_view


blog_main_link = 'https://www.themix.org.uk/news-and-research/blogs'


def get_blog_info(blog_main_link):
    big_list_name = []
    big_list_view = []

    for x in range(15):
        try:
            sub_link = blog_main_link + f'/page/{x+1}'
            soup = get_data(sub_link)

            name_list = []
            view_list = []

            results = soup.find_all('div', {'class': 'masonry-boxes__item'})
            for item in results:
                sub_link = item.find('a')['href']
                s = get_data(sub_link)
                name = clean_text(s.find('h1', {'class': 'label-blog'}).text)
                view = s.find("span", title=re.compile("Page Views")).text

                name_list.append(name)
                view_list.append(view)

            big_list_name.extend(name_list)
            big_list_view.extend(view_list)
        except:
            pass

    return big_list_name, big_list_view


cs_main_link = 'https://www.themix.org.uk/news-and-research/case-studies'


def get_cs_info(cs_main_link):
    big_list_name = []
    big_list_view = []

    for x in range(15):
        try:
            sub_link = cs_main_link + f'/page/{x+1}'
            soup = get_data(sub_link)

            name_list = []
            view_list = []

            results = soup.find_all('div', {'class': 'masonry-boxes__item'})
            for item in results:
                sub_link = item.find('a')['href']
                s = get_data(sub_link)
                name = clean_text(
                    s.find('h1', {'class': 'label-case_study'}).text)
                view = s.find("span", title=re.compile("Page Views")).text

                name_list.append(name)
                view_list.append(view)

            big_list_name.extend(name_list)
            big_list_view.extend(view_list)
        except:
            pass

    return big_list_name, big_list_view


research_link = 'https://www.themix.org.uk/news-and-research/research'


def get_research_title(research_link):
    soup = get_data(research_link)

    results = soup.find_all('div', {'class': 'campaigns__item-content'})
    name_list = []
    for item in results:
        name = clean_text(item.text)
        name_list.append(name)

    return name_list


#############################################################################
#                          Your Voice Tab
#############################################################################

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    # Remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)
    text = text.strip()
    return text


def clean_digits(text):
    return int(text.split()[1].replace(',', ''))


def get_data(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup


story_url = 'https://www.themix.org.uk/your-voices/submissions'


def get_story_info(story_url):
    big_title_list = []
    big_view_list = []

    for x in range(30):
        try:
            sublink = story_url + f'/page/{x+1}'
            soup = get_data(sub_link)
            results = soup.find_all('div', {'class': 'your-stories__content'})

            title_list = []
            view_list = []
            for item in results:
                link = item.find('a')['href']
                title = item.find(
                    'div', {'class': 'flag__body'}).find('a').text
                title = clean_text(title)
                s = get_data(link)
                view = s.find("span", title=re.compile("Page Views")).text
                view = clean_digits(view)
                title_list.append(title)
                view_list.append(view)

            big_title_list.extend(title_list)
            big_view_list.extend(view_list)
        except:
            pass

    return big_title_list, big_view_list
