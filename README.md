# 💬 WhatsApp Chat Analyzer

An interactive dashboard that analyzes any exported WhatsApp chat — cleaning raw chat text and extracting meaningful insights.

## 📊 Features

- Upload any WhatsApp chat export (.txt) — works with personal or group chats
- Automatic data cleaning (removes system messages, parses multi-line texts, filters junk senders)
- Most active participants ranking
- Daily message activity trend
- Hour-of-day and day-of-week activity patterns
- Most commonly used words (with stopword filtering)
- Key conversation insights

## 🛠️ Tech Stack

- Python, Streamlit, Pandas, Numpy, Plotly, Regex

## 📁 Data Source

- User-uploaded WhatsApp chat export (.txt) — no external dataset needed

## 🚀 How to Run

​```bash
pip install -r requirements.txt
streamlit run app.py
​```

## 💡 Key Features Explained

- Regex-based parsing of raw WhatsApp text format
- Custom data cleaning pipeline (removes system messages, junk senders)
- Word frequency analysis using Python's Counter

## 👤 Author

Hamza Iqbal — Data Science Portfolio Project