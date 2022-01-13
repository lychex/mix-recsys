import pandas as pd
import numpy as np
import datetime as dt
import re


def days_minutes(td):
    """convert timedelta into days and minutes"""
    return td.days, td.seconds//3600 * 60 + (td.seconds//60)%60

def day_gap(x):
    if abs(days_minutes(x)[0]) > 1:
        y = np.nan
    else:
        y = days_minutes(x)[1]
    return y

def clean_url_col(text):
    """Clean pages that are specific to a user"""
    # Ignore user searched page
    text = re.sub(r"(^\/search.*)", '', text)
    text = re.sub(r"(\?.*$)", '', text)
    text = re.sub('\/,', '/', text)
    text = re.sub(r"(^\/get-support\/find-local-services.*)", "/get-support/find-local-services", text)
    text = re.sub(r"(^\/post\/editdiscussion.*)", '/post/editdiscussion', text)
    text = re.sub(r"(^\/get-involved\/support-us.*)", '/get-involved/support-us', text)
    text = re.sub(r"(^.*course.*)", '/course', text)
    text = re.sub(r"(^.*mod.*)", '/mod', text)
    # Ignore user setting pages
    text = re.sub(r"(^\/profile.*)", '', text)
    text = re.sub(r"(^\/message.*)", '/message', text)
    text = re.sub(r"(^\/draft.*)", '', text)
    text = re.sub(r"(^\/login.*)", '', text)
    text = re.sub(r"(^\/user.*)", '', text)
    text = re.sub(r"(^.*forgotten-password.*)", "", text)
    
    return text


def create_feature(text):
    text = re.sub('^\/discussion.*', 'discussion', text)
    text = re.sub('^\/disussion.*', 'discussion', text)
    text = re.sub('^\/vanilla/discussion.*', 'discussion', text)
    text = re.sub('.*\/group-chat.*', 'chat', text)
    text = re.sub('^\/categories.*', 'community', text)
    text = re.sub('^\/activity.*', 'community', text)
    text = re.sub('^\/message.*', 'community', text)
    text = re.sub('^\/badge.*', 'community', text)
    text = re.sub('^\/post.*', 'community', text)
    text = re.sub('^\/report\/progress.*', 'community', text)
    
    text = re.sub('^\/get-support.*', 'support', text)
    text = re.sub('^\/brexit-euss-support.*', 'brexit support', text)
    text = re.sub('^\/coronavirus-support.*', 'coronavirus support', text)
    text = re.sub('^\/loneliness-support.*', 'loneliness support', text)
    text = re.sub('^\/brexit-eu-settlement-scheme-support.*', 'brexit support', text)
    text = re.sub('^\/young-carers.*', 'youngcarer support', text)
    text = re.sub('^\/bullying-support.*', 'support', text)
    text = re.sub('^\/self-harm-awareness-day.*', 'self-harm support', text)
    text = re.sub('^\/local.*', 'support', text)
    text = re.sub('^\/relationship-advice.*', 'relationship advice', text)
    text = re.sub('^\/services\/one-to-one-chat.*', 'chat', text)
    
    text = re.sub('.*.html$', 'article', text)
    text = re.sub('^\/mental-health.*', 'article', text)
    text = re.sub('^\/money.*', 'article', text)
    text = re.sub('^\/drink-and-drugs.*', 'article', text)
    text = re.sub('^\/travel-and-lifestyle.*', 'article', text)
    text = re.sub('^\/housing.*', 'article', text)
    text = re.sub('^\/work-and-study.*', 'article', text)
    text = re.sub('^\/your-body.*', 'article', text)
    text = re.sub('^\/crime-and-safety.*', 'article', text)
    text = re.sub('^\/coronavirus-and-mental-health.*', 'article', text)
    text = re.sub('^\/sex-and-relationships.*', 'article', text)
    text = re.sub('^\/tag.*', 'article', text)
    text = re.sub('^\/news-and-research.*', 'news', text)
    text = re.sub('^\/edit-profile', 'login', text)
    text = re.sub('^\/entry', 'login', text)
    text = re.sub('^\/log-in.*', 'login', text)
    text = re.sub('^.*login.*', 'login', text)
    text = re.sub('^\/sign-up.*', 'login', text)
    text = re.sub('^\/reset-password.*', 'login', text)
    text = re.sub('^\/admin.*', 'login', text)
    text = re.sub('^\/about-us.*', 'mixinfo', text)
    text = re.sub('^\/$', 'mixinfo', text)
    text = re.sub('^\/services\/counselling-services', 'mixinfo', text)
    text = re.sub('^\/partner\/counselling-from-the-mix', 'mixinfo', text)
    text = re.sub('^\/services\/telephone-counselling', 'mixinfo', text)
    text = re.sub('^\/services\/telephone-counselling', 'mixinfo', text)
    text = re.sub('^\/lib\/editor\/atto\/plugins.*', 'mixinfo', text)
    text = re.sub('^\/quartlerly-data-trends.*', 'mixinfo', text)
    text = re.sub('^\/brexit.*', 'mixinfo', text)
    text = re.sub('^\/trusted-information.*', 'mixinfo', text)
    
    text = re.sub('^\/mod.*', 'volunteer', text)
    text = re.sub('.*course.*', 'course', text)
    text = re.sub('^\/enrol.*', 'course', text)
    text = re.sub('^\/calendar.*', 'course', text)
    text = re.sub('^\/index.*', 'course', text)
    text = re.sub('^\/my\/.*', 'course', text)

    text = re.sub('^\/apps-and-tools.*', 'apps', text)
    text = re.sub('^\/home-truths.*', 'apps', text)
    text = re.sub('^\/get-involved.*', 'participation', text)
    text = re.sub('^\/get_involved.*', 'participation', text)
    text = re.sub('^\/body-and-soul-club.*', 'volunteer', text)
    text = re.sub('^\/digital-families.*', 'support', text)
    text = re.sub('.*digital families.*', 'support', text)
    text = re.sub('^\/your-voices.*', 'voices', text)
    text = re.sub('^\/bestof/everything.*', 'voices', text)
    
    return text

def get_pairs(article_dict):
    pairs = []

    for tag, names in article_dict.items():
        size = len(names)
        pairs.extend(list(zip([tag]*size, names)))

    # Remove empty values
    pairs = [t for t in pairs if t[1]!=''] 
    return pairs

def tag_mapper(search_text, pairs):
    for tag, name in pairs:
        if name == search_text:
            return tag

def clean_text(text):
    try:
        text = text.split('-')[0].strip()
        text = re.sub('[^a-zA-Z]', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove tags
        text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
        # Remove special characters and digits
        text = re.sub("(\\d|\\W)+", " ", text)
        text = text.strip()
    except:
        pass

    return text

