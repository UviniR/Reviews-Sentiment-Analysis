import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from wordcloud import WordCloud
from transformers import pipeline

# Load the pre-trained sentiment analysis model
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define the Streamlit app's user interface
# Set page title and favicon
st.set_page_config(page_title="Review Analysis", page_icon=":smiley:")

# Add image and heading
st.image("Header.png", use_column_width=True)

file = st.file_uploader(" ",label_visibility='collapsed',type=["csv"])

# Define the app's functionality
if file is not None:
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(file)
    st.markdown(f"<h5 style='font-family: sans-serif;margin-top:40px'>Total reviews: {len(df)-1} </h5>", unsafe_allow_html=True)
    
    # Write the total number of records
    st.markdown(
        f'<div style="background-color: #4AA6DD; color: #ffffff; padding: 6px; font-size: 16px; font-weight: bold; text-align: center; border-radius: 1rem;margin-top: 10px"> Distribution of Reviews </div>',
        unsafe_allow_html=True
    )

    # Apply the sentiment analysis model to each review and store the results in a new column
    df["Sentiment"] = df["Review"].apply(lambda x: sentiment_model(x)[0]["label"])

    # Generate pie chart
    # Define custom colors
    colors = ['#30C3C4', '#D1DDDE']

    # Generate pie chart
    sentiment_counts = df["Sentiment"].value_counts()
    fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index,
                 color_discrete_sequence=colors)
    st.plotly_chart(fig, use_container_width=True)

    # Create word clouds for positive and negative reviews
    positive_reviews = " ".join(df[df["Sentiment"] == "POSITIVE"]["Review"].tolist())
    negative_reviews = " ".join(df[df["Sentiment"] == "NEGATIVE"]["Review"].tolist())

    # Center-align the word clouds
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f'<div style="background-color: #4AA6DD; color: #ffffff; padding: 6px; font-size: 16px; font-weight: bold; text-align: center; margin-bottom: 40px; border-radius: 1rem">Positive Reviews</div>',
            unsafe_allow_html=True
        )
        wc_pos = WordCloud(width=800, height=600, background_color="white", colormap="winter").generate(positive_reviews)
        st.image(wc_pos.to_array(),use_column_width=True)
    
    with col2:
        st.markdown(
            f'<div style="background-color: #4AA6DD; color: #ffffff; padding: 6px; font-size: 16px; font-weight: bold; text-align: center; margin-bottom: 40px;border-radius: 1rem">Negative Reviews</div>',
            unsafe_allow_html=True
        )
        wc_neg = WordCloud(width=800, height=600, background_color="white", colormap="winter").generate(negative_reviews)
        st.image(wc_neg.to_array(),use_column_width=True)

    # Display the sentiment of each review as cards
    st.markdown(
        f'<div style="background-color: #4AA6DD; color: #ffffff; padding: 6px; font-size: 16px; font-weight: bold; text-align: center; margin-top: 60px; border-radius: 1rem"> Reviews in depth </div>',
        unsafe_allow_html=True
    )

    # Add the selectbox to filter sentiments
    filter_sentiment = st.selectbox("", ["ALL", "POSITIVE", "NEGATIVE"])
    
    # Filter the dataframe based on the selected sentiment
    if filter_sentiment != "ALL":
        df = df[df['Sentiment'] == filter_sentiment]
    
    # Set the max number of rows to display at a time
    max_rows = 15
    
    # Create HTML table with no border and centered text
    table_html = (df.style
                  .set_properties(**{'text-align': 'left','font-size': '14px'})
                  .set_table_styles([{'selector': 'th', 'props': [('border', '0px')]},
                                     {'selector': 'td', 'props': [('border', '0px')]}])
                  .set_table_attributes('style="position: sticky; top: 0;"')
                  .to_html(index=False, escape=False))
    
    # Wrap the table inside a div with a fixed height and scrollable content
    st.write(f'<div style="height: {max_rows*30}px; overflow-y: scroll;">{table_html}</div>', unsafe_allow_html=True,header=True,sticky_header=True)

    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    
    csv = convert_df(df)
    
    # Add some space between the download button and the table
    st.write("<br><br>", unsafe_allow_html=True)
    
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Review Sentiments.csv'
    )
st.write("<br>", unsafe_allow_html=True)
st.caption('<div style="text-align:center; background-color:#CFEDFF;padding: 6px">crafted with ❤️</div>', unsafe_allow_html=True)
