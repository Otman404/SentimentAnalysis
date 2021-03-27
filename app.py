import streamlit as st
import tweepy
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import pickle
import twitter_creds
# import GetOldTweets3 as got
import datetime
import base64
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from string import punctuation
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('stopwords')

def connect_to_twitter_OAuth():
    auth = tweepy.OAuthHandler(twitter_creds.CONSUMER_KEY, twitter_creds.CONSUMER_SECRET)
    auth.set_access_token(twitter_creds.ACCESS_TOKEN, twitter_creds.ACCESS_SECRET)
    api = tweepy.API(auth)
    return api

api = connect_to_twitter_OAuth()

@st.cache(persist=True,show_spinner=False)
def tweets_to_data_frame(searchTerm,nbrOfTweets):

    tweets = api.user_timeline(screen_name=searchTerm, count=nbrOfTweets,tweet_mode='extended')
    df = pd.DataFrame(data=[tweet.full_text for tweet in tweets], columns=['tweets'])

    df['id'] = np.array([tweet.id for tweet in tweets])
    df['len'] = np.array([len(tweet.full_text) for tweet in tweets])
    df['date'] = np.array([tweet.created_at for tweet in tweets])
    df['device'] = np.array([tweet.source for tweet in tweets])
    df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
    df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])
    df['link'] = np.array([f"https://twitter.com/_/status/{tweet.id}" for tweet in tweets])
    return df

def clean_text(text):
    import re
    #lower text
    text = text.lower()
    #removing mentions
    text = re.sub("@(\w+)","",text)
    #removing hashtags
    text = re.sub(r"#(\w+)","",text)
    #removing RT
    text = text.replace("RT","")
    #decontract
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    #removing numbers and special characters
    text = re.sub("[^A-Za-z ]+","",text)
    #remove links
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)

    
    #tokenize text
    text_tokenized = nltk.word_tokenize(text,language='english')
    
    #removing stopwords
    negtion_words = ['no','not','none','no','one','nobody','nothing','neither','nowhere','never','doesn\'t','isn\'t','wasn\'t','shouldn\'t','wouldn\'t','couldn\'t','won\'t','can\'t','don\'t']
    ## keeping negation words since it affects the sentiment of a text
    my_stopwords = set(negtion_words) ^ set(stopwords.words('english'))
    
    text_tokenized = [w for w in text_tokenized if w not in my_stopwords]
    
    #lemmetizing text
    lemmatizer = WordNetLemmatizer()
    text_lemmatized = [lemmatizer.lemmatize(w) for w in text_tokenized]
    
    #removing stopwords
    return ' '.join(text_lemmatized)

# @st.cache(persist=True,show_spinner=False)
# def get_tweets(keyword, startdate, enddate, maxtweet):
#     tweetCriteria = got.manager.TweetCriteria().setQuerySearch(keyword)\
#                                             .setTopTweets(True)\
#                                             .setSince(startdate)\
#                                             .setUntil(enddate)\
#                                             .setMaxTweets(maxtweet)
                                            
#                                             # .setNear(location)\
#     tweet = got.manager.TweetManager.getTweets(tweetCriteria)
    
#     text_tweets = [[tw.username,
#                 tw.text,
#                 tw.date,
#                 tw.retweets,
#                 tw.favorites,
#                 tw.mentions,
#                 tw.hashtags,
#                 tw.geo] for tw in tweet]    
    
#     df= pd.DataFrame(text_tweets, columns = ['User', 'Text', 'Date', 'Likes', 'Retweets', 'Mentions','Hashtags', 'Geolocation'])
    
#     return df

@st.cache(persist=True,show_spinner=False,allow_output_mutation=True)
def scrap_reviews(asin_no,max_page_numbers=3):
    """
    asin_no: Each product have a unique number, ex: B07ZQRMWVB
    max_page_numbers: reviews have pagination, we specify number of pages we want to scrap, else this might take for ever if a product have a lot of reviews.
    """
    import requests
    from bs4 import BeautifulSoup
    from fake_useragent import UserAgent
    import time
    # import pandas as pd
    ua = UserAgent()
    base_url = 'https://www.amazon.com/product-reviews/{asin_no}/reviewerType=all_reviews&?pageNumber={page_number}'
    page_number = 1
    customer_reviews = []

    review_title = []
    review_text = []
    while page_number<max_page_numbers:
        s = requests.Session()
        r = s.get(base_url.format(asin_no=asin_no, page_number=page_number), headers={
            'Referer': f'https://www.amazon.com/dp/{asin_no}/',
            'User-Agent': ua.random,
        })
        html = r.text
        soup = BeautifulSoup(html, 'html.parser')
        product_title = soup.select(".a-size-large > a:nth-child(1)")[0].text
        customer_review_elements = soup.select('*[id*="customer_review-"]')
        for customer_review_element in customer_review_elements:
            review_title.append(customer_review_element.select_one('.review-title').text.strip())
            review_text.append(customer_review_element.select_one('.review-text').text.strip())

        if len(customer_review_elements) > 0:
            page_number += 1
            time.sleep(1)
        else:
            break
    customer_reviews = pd.DataFrame({'title': review_title, 'text': review_text})
    return product_title,customer_reviews


def get_table_download_link(df,csv_name):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{csv_name}">Download csv file</a>'
    return href


def show_wordcloud(data, title = None):
                    wordcloud = WordCloud(
                        background_color='black',
                        stopwords=stopwords.words('english'),
                        max_words=200,
                        max_font_size=40, 
                        scale=3,
                        random_state=1 # chosen at random by flipping a coin; it was heads
                ).generate(str(data))

                    fig = plt.figure(1, figsize=(15, 15))
                    plt.axis('off')
                    if title: 
                        fig.suptitle(title, fontsize=20)
                        fig.subplots_adjust(top=2.3)

                    plt.imshow(wordcloud)
                    st.pyplot(fig)



st.sidebar.title("Sentiment Analysis")
st.sidebar.markdown(":smile: vs :rage:")
st.sidebar.markdown("This is a project I made as an oppertunity to learn about Natural Language Processing, Sentiment Analysis, and how to handle text data.")
st.sidebar.markdown("#### You can find this project here on [Github](https://github.com/Otman404/SentimentAnalysis)")
# st.title('NLP: Sentiment Analysis')



action = st.sidebar.radio('Choose your action:',('Analyze Text','Analyze Tweets','Analyze Product Reviews'))



with open('models/twitter_svm_model.pkl', 'rb') as file:  
    twitter_model = pickle.load(file)



if action == 'Analyze Text':
    st.title('Analyze Text')
    text = st.text_area('Type something...')
    if st.button('Analyze'):
        with st.spinner('Analyzing sentiment ...'):
            if not text:
                st.write('Textfield is empty')
            else:
                sentiment = twitter_model.predict([clean_text(text)])
                pos = "<span style='color:green;font-weight:500'>Positive :smile:</span>" 
                neg = "<span style='color:red;font-weight:500'>Negative :rage:</span>"
                # st.markdown(f"## This text is {'Positive :smile:' if sentiment else 'Negative :rage:'}")
                st.markdown(f"## This text is {pos if sentiment else neg}",unsafe_allow_html=True)

elif action == 'Analyze Tweets':
    st.title('Analyze Tweets')

    keyword = st.text_input('Keyword',value="Twitter")

    nbrOfTweets = st.number_input('Number of Tweets (< 200)',value=10)
    if st.button('Get Tweets'):
        with st.spinner('Fetching Tweets ...'):
            try:
                st.write(f'Showing results about: "{keyword}"...')
                df = tweets_to_data_frame(keyword, nbrOfTweets)
                df['cleaned_tweets'] = df['tweets'].apply(clean_text)
                preds = twitter_model.predict(df['cleaned_tweets'])
                df['Sentiment'] = preds
                fig = go.Figure()

                likes = pd.Series(data=df['likes'].values,index=df.date) # use 'Likes' instead of 'likes' for GetOldTweets3
                retweets = pd.Series(data=df['retweets'].values,index=df.date) # use 'Retweets' instead of 'retweets' for GetOldTweets3



                fig.add_trace(go.Scatter(x=df.date, y=df.likes,name="likes"))
                fig.add_trace(go.Scatter(x=df.date, y=df.retweets,name="retweets"))
                fig.update_layout(title='Likes vs Retweets')
                st.plotly_chart(fig,use_container_width=True)

                labeled_sentiment = []
                for s in df['Sentiment']:
                    if s:
                        labeled_sentiment.append("Positive")
                    else:
                        labeled_sentiment.append("Negative")
                df['Sentiment_Label'] = labeled_sentiment

                fig = px.pie(df, names='Sentiment_Label')
                fig.update_layout(title='Positive Tweets vs Negative Tweets')

                st.plotly_chart(fig)

                st.markdown('### Most Frequent Words')
                # stopwords = set(STOPWORDS)
                show_wordcloud(df["tweets"])

                st.markdown('### Distribution of word lengths')

                # Distribution of word lengths
                word_count = [len(word_tokenize(t, "english")) for t in df['tweets']]
                plt.figure()    
                plt.hist(word_count,bins=np.arange(0,max(word_count),10))
                plt.xlabel('Tweets Lenghts')
                plt.ylabel('Number of Tweets')
                st.pyplot(plt)


                st.markdown(get_table_download_link(df,'tweets.csv'), unsafe_allow_html=True)
                show_data = st.checkbox('Show Tweets')
                if show_data:
                    st.write(df)


            except Exception as ex:

                st.markdown("<span style='color:red'>Please search another keyword</span>",unsafe_allow_html=True)
                st.markdown(f"<span style='color:red'>{ex}</span>",unsafe_allow_html=True)
else:
    st.title('Amazon electronic Product Reviews')
    with open('models/amazon_lr_model.pkl', 'rb') as m:  
       reviews_model = pickle.load(m)
    asin_no = st.text_input('Product Asin',value="B0881ZF6WP")
    max_page_numbers = st.number_input('Max Page numbers',value=3)
    if st.button('Get Reviews'):
        with st.spinner('Fetching Product Reviews ...'):
            try:
                product_title,reviews = scrap_reviews(asin_no,max_page_numbers)
                st.markdown(f"<strong>Showing results about:</strong> {product_title}",unsafe_allow_html=True)
                st.markdown(f"<strong>Number of reviews:</strong> {reviews.shape[0]}",unsafe_allow_html=True)
                reviews['clean_reviews'] = reviews['text'].apply(clean_text)
                preds = reviews_model.predict(reviews['clean_reviews'])
                reviews['Sentiment'] = preds
                labeled_sentiment = []
                for s in reviews['Sentiment']:
                    if s == 1:
                        labeled_sentiment.append("Positive")
                    elif s == -1:
                        labeled_sentiment.append("Negative")
                    else:
                        labeled_sentiment.append("Neutral")
                reviews['Sentiment_Label'] = labeled_sentiment

                fig = px.pie(reviews, names='Sentiment_Label')
                fig.update_layout(title='Positive Reviews vs Negative Reviews')
                st.plotly_chart(fig)

                st.markdown('### Most Frequent Words')
                # stopwords = set(STOPWORDS)
                show_wordcloud(reviews["text"])

                st.markdown('### Distribution of word lengths')

                # Distribution of word lengths
                word_count = [len(word_tokenize(t, "english")) for t in reviews["text"]]
                plt.figure()    
                plt.hist(word_count,bins=np.arange(0,max(word_count),10))
                plt.xlabel('Reviews Lenghts')
                plt.ylabel('Number of Reviews')
                st.pyplot(plt)
                st.markdown(get_table_download_link(reviews,'reviews.csv'), unsafe_allow_html=True)

            except Exception as ex:
                st.markdown(f"<span style='color:red'>{ex}</span>",unsafe_allow_html=True)







