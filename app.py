# WhatsApp Chat Analyzer
# Data: User's exported WhatsApp chat (.txt file)

import streamlit as st
import pandas as pd
import plotly.express as px
import re
import numpy as np
from collections import Counter

st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide"
)

st.title("💬 WhatsApp Chat Analyzer")
st.markdown("Upload your exported WhatsApp chat and get insights into the conversation")

# ── FILE UPLOAD ──────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your chat .txt file", type="txt")

if uploaded_file is None:
    st.info("Please upload a WhatsApp chat export (.txt) to begin")
    st.stop()

# read the raw text file
raw_text = uploaded_file.read().decode("utf-8")

with st.expander("Show Raw Chat Text (first 1000 characters)"):
    st.text(raw_text[:1000])

# ── REGEX PATTERN TO PARSE MESSAGES ────────────────────
# matches: 3/8/25, 8:17 AM - Sender: Message
pattern = r'(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}\s?[APap][Mm]) - (.*?): (.*)'

messages = []

for line in raw_text.split('\n'):
    match = re.match(pattern, line)
    if match:
        date, time, sender, message = match.groups()
        messages.append({
            "date": date,
            "time": time,
            "sender": sender,
            "message": message
        })
    else:
        # if line doesn't match (continuation of previous message), append to last message
        if messages and line.strip():
            messages[-1]["message"] += " " + line.strip()

df = pd.DataFrame(messages)

# ── DATA CLEANING ──────────────────────────────────────
# convert date column to proper datetime
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%y %I:%M %p')

# remove system messages (group created, security code changed, etc.)
system_keywords = ['created group', 'changed the subject', 'added', 'left', 
                    'security code', 'joined using', 'end-to-end encrypted', 'removed']

df['is_system'] = df['message'].str.contains('|'.join(system_keywords), case=False, na=False)
df = df[~df['is_system']].copy()

# remove media omitted messages for word analysis (keep for count though)
df['is_media'] = df['message'].str.contains('Media omitted', case=False, na=False)

# extract day, hour, month for time-based analysis
df['day_name'] = df['datetime'].dt.day_name()
df['hour'] = df['datetime'].dt.hour
df['date_only'] = df['datetime'].dt.date

# word count per message
df['word_count'] = df['message'].apply(lambda x: len(x.split()))

# remove rows where sender is just a dot or empty/junk
df = df[df['sender'].str.len() > 2]

st.markdown("---")

# ── OVERVIEW METRICS ───────────────────────────────────
st.subheader("Chat Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Messages", len(df))
with col2:
    st.metric("Total Participants", df['sender'].nunique())
with col3:
    total_words = np.sum(df['word_count'])
    st.metric("Total Words", f"{total_words:,}")
with col4:
    media_count = df['is_media'].sum()
    st.metric("Media Shared", media_count)

st.markdown("---")

with st.expander("Show Cleaned Data"):
    st.dataframe(df[['datetime', 'sender', 'message']].head(50))
    st.write("Shape after cleaning:", df.shape)

st.markdown("---")

# ── MOST ACTIVE PARTICIPANTS ────────────────────────────
st.subheader("Most Active Participants")

# count messages per sender
sender_counts = df['sender'].value_counts().head(10).reset_index()
sender_counts.columns = ['Sender', 'Messages']

# force Sender column to be treated as text, not numbers
sender_counts['Sender'] = sender_counts['Sender'].astype(str)

fig1 = px.bar(
    sender_counts,
    x="Messages",
    y="Sender",
    orientation="h",
    title="Top 10 Most Active Members",
    color="Messages",
    color_continuous_scale="Greens"
)
fig1.update_layout(
    yaxis={'categoryorder': 'total ascending', 'type': 'category'}
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ── ACTIVITY OVER TIME ──────────────────────────────────
st.subheader("Messages Over Time")

daily_counts = df.groupby('date_only').size().reset_index(name='count')

fig2 = px.line(
    daily_counts,
    x="date_only",
    y="count",
    title="Daily Message Activity",
    markers=True
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── ACTIVITY BY HOUR ─────────────────────────────────────
st.subheader("What Time of Day is the Group Most Active?")

hourly_counts = df.groupby('hour').size().reset_index(name='count')

fig3 = px.bar(
    hourly_counts,
    x="hour",
    y="count",
    title="Messages by Hour of Day (24-hour format)",
    color="count",
    color_continuous_scale="Purples"
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── ACTIVITY BY DAY OF WEEK ─────────────────────────────
st.subheader("Most Active Days")

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = df['day_name'].value_counts().reindex(day_order).reset_index()
day_counts.columns = ['Day', 'Messages']

fig4 = px.bar(
    day_counts,
    x="Day",
    y="Messages",
    title="Messages by Day of Week",
    color="Messages",
    color_continuous_scale="Blues"
)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ── MOST COMMON WORDS ───────────────────────────────────
st.subheader("Most Common Words")

# combine all messages (excluding media) into one text blob
all_text = ' '.join(df[~df['is_media']]['message'].astype(str))

# clean text - keep only words, lowercase
words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())

# remove common stopwords
stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 
             'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 
             'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 
             'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 
             'use', 'that', 'with', 'have', 'this', 'will', 'your', 'from',
             'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time'}

filtered_words = [w for w in words if w not in stopwords]

word_freq = Counter(filtered_words).most_common(15)
word_freq_df = pd.DataFrame(word_freq, columns=['Word', 'Count'])

fig5 = px.bar(
    word_freq_df,
    x="Count",
    y="Word",
    orientation="h",
    title="Top 15 Most Used Words",
    color="Count",
    color_continuous_scale="Oranges"
)
fig5.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# ── KEY INSIGHTS ─────────────────────────────────────────
st.subheader("Key Insights")

col1, col2 = st.columns(2)

with col1:
    most_active = sender_counts.iloc[0]
    st.success(f"💬 Most active: **{most_active['Sender']}** with **{most_active['Messages']}** messages")

    busiest_hour = hourly_counts.loc[hourly_counts['count'].idxmax(), 'hour']
    st.info(f"🕒 Busiest hour: **{busiest_hour}:00**")

with col2:
    busiest_day = day_counts.loc[day_counts['Messages'].idxmax(), 'Day']
    st.warning(f"📅 Most active day: **{busiest_day}**")

    avg_words = np.mean(df['word_count'])
    st.error(f"📝 Average words per message: **{avg_words:.1f}**")

st.markdown("---")
st.markdown("**Data Source:** Exported WhatsApp Chat (.txt)")
st.markdown("**Built with:** Python, Streamlit, Pandas, Numpy, Plotly, Regex")