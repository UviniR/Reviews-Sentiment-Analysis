# import essential libraraies
import streamlit as st
import pandas as pd
from transformers import pipeline #for pre-trained model
# for graphing
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from wordcloud import WordCloud

# Load the pre-trained model for sentiment analysis 
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define user interface

# Set page title
st.set_page_config(page_title="Hotel Reviews Sentiment", page_icon=":hotel:",layout='wide')
# Add a header
st.image("Header.png", use_column_width=True)
# Define the format of the file to be uploaded
st.write("<br>", unsafe_allow_html=True)
file_format_link = "https://drive.google.com/file/d/1B6Glpna9kZsakgjpWStfhbxI20EoGsnv/view?usp=sharing"
message = f"⚠️ㅤPlease stick to the given layout when uploading the file. You can download the sample file layout from [here]({file_format_link})."
st.write(message)
# Uploading the file
file = st.file_uploader("",type=["csv"])

if file is not None:
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(file)
    # Print total number of reviews to analyse
    st.markdown(f"<h5 style='font-family: sans-serif;margin-top:40px'>Total reviews: {len(df)} </h5>", unsafe_allow_html=True)
    
    st.markdown(
        f'<div style="background-color: #4AA6DD; color: #ffffff; padding: 6px; font-size: 20px; font-weight: bold; text-align: center; border-radius: 1rem;margin-top: 10px"> Distribution of Reviews </div>',
        unsafe_allow_html=True
    )

    # Apply the sentiment analysis model
    df["Sentiment"] = df["Review"].apply(lambda x: sentiment_model(x)[0]["label"])
    
    # Building the dashboard
    
    # Generate pie chart for sentiment distribution
    colors = ['#30C3C4', '#D1DDDE']
    sentiment_counts = df["Sentiment"].value_counts()
    fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index,
                 color_discrete_sequence=colors)
    st.plotly_chart(fig, use_container_width=True)

    # Create word clouds for positive and negative reviews
    positive_reviews = " ".join(df[df["Sentiment"] == "POSITIVE"]["Review"].tolist())
    negative_reviews = " ".join(df[df["Sentiment"] == "NEGATIVE"]["Review"].tolist())
    # Diplay wordcloud in two columns 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div style="background-color: #4AA6DD; color: #ffffff; padding: 6px; font-size: 20px; font-weight: bold; text-align: center; margin-bottom: 40px; border-radius: 1rem">Positive Reviews</div>',
            unsafe_allow_html=True
        )
        wc_pos = WordCloud(width=800, height=600, background_color="white", colormap="winter").generate(positive_reviews)
        st.image(wc_pos.to_array(),use_column_width=True)
    with col2:
        st.markdown(
            f'<div style="background-color: #4AA6DD; color: #ffffff; padding: 6px; font-size: 20px; font-weight: bold; text-align: center; margin-bottom: 40px;border-radius: 1rem">Negative Reviews</div>',
            unsafe_allow_html=True
        )
        wc_neg = WordCloud(width=800, height=600, background_color="white", colormap="winter").generate(negative_reviews)
        st.image(wc_neg.to_array(),use_column_width=True)

    # Display the sentiment of each review as a dataframe
    st.markdown(
        f'<div style="background-color: #4AA6DD; color: #ffffff; padding: 6px; font-size: 20px; font-weight: bold; text-align: center; margin-top: 60px; border-radius: 1rem"> Reviews in depth </div>',
        unsafe_allow_html=True
    )
    # Add a filter for sentiments
    filter_sentiment = st.selectbox("", ["ALL", "POSITIVE", "NEGATIVE"])
    # Filter the dataframe
    if filter_sentiment != "ALL":
        df = df[df['Sentiment'] == filter_sentiment]
    # Max number of rows to display at a time
    max_rows = 10
    
    # Table generation
    table_html = df.style.set_table_styles([{'selector': 'th', 'props': [('border', '0px')]},
                                             {'selector': 'td', 'props': [('border', '0px')]}]).render()
    
       # --- This section should be replaced for the above table generation when deploying the model on Hugging Face Space ---
#     table_html = (df.style
#                   .set_properties(**{'text-align': 'left','font-size': '14px'})
#                   .set_table_styles([{'selector': 'th', 'props': [('border', '0px')]},
#                                      {'selector': 'td', 'props': [('border', '0px')]}])
#                   .set_table_attributes('style="position: sticky; top: 0;"')
#                   .to_html(index=False, escape=False))
    
    # Scrollable content
    st.write(f'<div style="height: {max_rows*30}px; overflow-y: scroll;">{table_html}</div>', unsafe_allow_html=True,header=True,sticky_header=True)

    # save output as csv
    def convert_df(df):
        return df.to_csv().encode('utf-8')
    csv = convert_df(df)
    
    # Download button
    st.write("<br>", unsafe_allow_html=True)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Review Sentiments.csv'
    )
    
# Footnote
st.write("<br>", unsafe_allow_html=True)
st.write('<div style="text-align:center; color:#52565E; background-color:#CFEDFF;padding: 6px;font-size:14px;">crafted with ❤️</div>', unsafe_allow_html=True)

    # --- This section should be replaced for the above table generation when deploying the model on Hugging Face Space ---
#st.caption('<div style="text-align:center; background-color:#CFEDFF;padding: 6px">crafted with ❤️</div>', unsafe_allow_html=True)

# --- End of the code ---
