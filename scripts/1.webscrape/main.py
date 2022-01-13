import sys
import json



sys.path.append('./scripts/1.webScrape/modules')
# from data import *
from scrape import *


#=====================================================================================
# Scrape data from The Mix web 
#=====================================================================================

# Print the major tags
url = 'https://www.themix.org.uk/'

soup = get_data(url)
div_list = get_div_list(soup)
# Money page is different from other pages, modify it to make it work
div_list[5] = div_list[5] + '/benefits'
# div_list contains pages that are all article pages
print(div_list)


# Create ARTICLE catalog
all_article_topics = get_all_topics(div_list)

article_topic_catalog = dict()
div_list = get_div_list(soup)
for topic_title, topic_link in all_article_topics:
    article_topic_catalog[topic_title] = get_topic_detail(topic_link)


with open('./databases/static/article.txt', 'w') as file:
     file.write(json.dumps(article_topic_catalog)) 

# We can repeat the process to generate tags for other categories

# Create SUPPORT catalog
forum_url = 'https://community.themix.org.uk/'
discuss_board = get_categories(forum_url)[0]
discuss_board.append("recent discussion")
discuss_board

group_chat = ['group chat', 'community quiz', 'watch club', 'young carer', 
              'general chat', 'support chat', 'support circle', 'expert chat']
speak_to_team = ['speak to our team', 'helpline', 'email us', 'one to one chat',  'counselling', 'crisis messenger']
find_local_service = ['find local service']

support_dict = dict()

support_dict['discuss board'] = discuss_board
support_dict['group chat'] = group_chat
support_dict['speak to team'] = speak_to_team
support_dict['find local service'] = find_local_service


with open('./databases/static/support.txt', 'w') as file:
     file.write(json.dumps(support_dict)) 


