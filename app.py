# WhatsApp Chat Analyzer
# Data: User's exported WhatsApp chat (.txt file)

import streamlit as st
import pandas as pd
import plotly.express as px
import re
import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression

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
        if messages and line.strip():
            messages[-1]["message"] += " " + line.strip()

df = pd.DataFrame(messages)

# ── DATA CLEANING ──────────────────────────────────────
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%m/%d/%y %I:%M %p')

system_keywords = ['created group', 'changed the subject', 'added', 'left', 
                    'security code', 'joined using', 'end-to-end encrypted', 'removed']

df['is_system'] = df['message'].str.contains('|'.join(system_keywords), case=False, na=False)
df = df[~df['is_system']].copy()

df['is_media'] = df['message'].str.contains('Media omitted', case=False, na=False)

df['day_name'] = df['datetime'].dt.day_name()
df['hour'] = df['datetime'].dt.hour
df['date_only'] = df['datetime'].dt.date

df['word_count'] = df['message'].apply(lambda x: len(x.split()))

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

sender_counts = df['sender'].value_counts().head(10).reset_index()
sender_counts.columns = ['Sender', 'Messages']
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

# ── ML: FUTURE ACTIVITY PREDICTOR ───────────────────────
st.subheader("🔮 Predict Future Group Activity")
st.markdown("Using Linear Regression to predict how many messages will be sent on a future day")

# prepare data - convert dates into simple day numbers (day 1, day 2, day 3...)
activity_df = daily_counts.copy()
activity_df['date_only'] = pd.to_datetime(activity_df['date_only'])
activity_df = activity_df.sort_values('date_only')
activity_df['day_number'] = (activity_df['date_only'] - activity_df['date_only'].min()).dt.days

# X = day number (input), y = message count (what we predict)
X = activity_df[['day_number']]
y = activity_df['count']

# train the model
activity_model = LinearRegression()
activity_model.fit(X, y)

# let user pick how many days into the future to predict
days_ahead = st.slider("Predict activity this many days from the last message", 1, 30, 7)

future_day_number = activity_df['day_number'].max() + days_ahead
predicted_messages = activity_model.predict([[future_day_number]])[0]
predicted_messages = max(0, predicted_messages)  # messages can't be negative

future_date = activity_df['date_only'].max() + pd.Timedelta(days=days_ahead)

st.metric(
    f"Predicted messages on {future_date.strftime('%d %b %Y')}",
    f"{predicted_messages:.0f} messages"
)

st.info("⚠️ Note: This is a simple linear trend prediction based on overall activity pattern. Real group activity can spike or drop suddenly based on events, so treat this as a rough estimate.")

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

all_text = ' '.join(df[~df['is_media']]['message'].astype(str))

words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())

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
st.markdown("**Built with:** Python, Streamlit, Pandas, Numpy, Plotly, Regex, Scikit-learn")
