# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# Goal: Collect Expertise on a Topic that's relevant to your business application

# YOUTUBE API SETUP ------------------------------------------------
# 1. Go to https://console.developers.google.com/
# 2. Create a new project
# 3. Enable the YouTube Data API v3
# 4. Create credentials
# 5. Place the credentials in a file called credentials.yml formatted as follows:
# youtube: 'YOUR_API_KEY'

# 1.0 IMPORTS 

from langchain_community.document_loaders import YoutubeLoader

from googleapiclient.discovery import build

import yaml
import pandas as pd


# 2.0 YOUTUBE API KEY SETUP 

PATH_CREDENTIALS = '../credentials.yml'

YOUTUBE_API_KEY = yaml.safe_load(open(PATH_CREDENTIALS))['youtube'] 

# 3.0 VIDEO TRANSCRIPT SCRAPING FUNCTIONS

def search_videos(topic, api_key, max_results=20):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.search().list(
        q=topic,
        part='id,snippet',
        maxResults=max_results,
        type='video'
    )
    response = request.execute()
    video_ids = [item['id']['videoId'] for item in response['items']]
    return video_ids

def load_video(video_id):
    url = f'https://www.youtube.com/watch?v={video_id}'
    loader = YoutubeLoader.from_youtube_url(
        url, 
        add_video_info=True,
    )
    doc = loader.load()[0]
    doc_df = pd.DataFrame([doc.metadata])
    doc_df['video_url'] = url
    doc_df['page_content'] = doc.page_content
    return doc_df


# 4.0 SCRAPE YOUTUBE VIDEOS TRANSCRIPTS

TOPIC = "Social Media Brand Strategy Tips"

video_ids = search_videos(TOPIC, YOUTUBE_API_KEY, max_results=10)

# * Scrape the video metadata and page content
videos = []
for video_id in video_ids:
    video = load_video(video_id)
    
    videos.append(video)
   
videos_df = pd.concat(videos, ignore_index=True)
videos_df

# * Store the video transcripts in a CSV File
videos_df.to_csv('youtube_videos.csv', index=False)
