import os
from flask import Flask, render_template, request, send_from_directory
from dotenv import load_dotenv
import spacy
import trafilatura
from newspaper import Article
from fpdf import FPDF
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob

# Groq LLM
from groq import Groq

# ------------------ Setup ------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")

# Ensure directories exist
os.makedirs("static/pdfs", exist_ok=True)
os.makedirs("static/images", exist_ok=True)


# ------------------ Article Extraction ------------------
def extract_article(url):
    """Try trafilatura first, fallback to newspaper3k."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(downloaded)
        if text and len(text.split()) > 50:
            return text

    article = Article(url)
    article.download(timeout=20)
    article.parse()
    return article.text


# ------------------ Groq Summarizer ------------------
def generate_summary(article_text):
    """Generate detailed summary using Groq LLM."""
    prompt = f"""
You are a professional journalist. Summarize and expand this news article into a detailed, easy-to-read report 
with background, key points, and analysis. Do NOT include captions.

Article:
{article_text}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content


# ------------------ Visual Generation ------------------
def generate_combined_visuals(text, output_path="static/images/combined.png"):
    """Generate word cloud, keyword bar chart, pie chart, and sentiment bar chart on a single image."""
    nlp_doc = nlp(text.lower())
    words = [token.text for token in nlp_doc if token.is_alpha and not token.is_stop]
    freq = Counter(words).most_common(10)

    # Sentiment analysis
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    pos = max(sentiment, 0)
    neg = max(-sentiment, 0)
    neu = 1 - abs(sentiment)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Word Cloud
    wc_text = ' '.join(words)
    wc = WordCloud(width=400, height=300, background_color='white').generate(wc_text)
    axs[0,0].imshow(wc, interpolation='bilinear')
    axs[0,0].axis('off')
    axs[0,0].set_title("Word Cloud")

    # Bar chart
    if freq:
        kw, counts = zip(*freq)
        axs[0,1].bar(kw, counts, color='skyblue')
        axs[0,1].set_title("Top Keywords")
        axs[0,1].tick_params(axis='x', rotation=45)

    # Pie chart
    if freq:
        axs[1,0].pie(counts, labels=kw, autopct='%1.1f%%', startangle=140)
        axs[1,0].set_title("Keyword Distribution")

    # Sentiment bar
    axs[1,1].bar(["Positive", "Neutral", "Negative"], [pos, neu, neg], color=['green','grey','red'])
    axs[1,1].set_title("Overall Sentiment")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


# ------------------ PDF Generation with Styled Headings ------------------
def generate_pdf(summary, visual_path=None, filename="static/pdfs/news.pdf"):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", size=12)
    
    lines = summary.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(4)
            continue
        
        # Remove markdown-style asterisks
        line = line.replace("**", "")
        
        # Bold headings
        if line.isupper() or line.startswith("HEADLINE") or line.startswith("KEY POINTS") or line.startswith("ANALYSIS") or line.startswith("BACKGROUND"):
            pdf.set_font("Arial", 'B', 14)
            pdf.multi_cell(0, 8, line)
            pdf.ln(2)
        # Italic subheadings (points)
        elif line.startswith("-") or line.startswith("•") or (line[0].isdigit() and line[1] in ['.', ')']):
            pdf.set_font("Arial", 'I', 12)
            pdf.multi_cell(0, 8, line)
        else:
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, line)
    
    # Insert combined visual with title
    if visual_path and os.path.exists(visual_path):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Visuals", ln=True, align='C')  # centered title
        pdf.ln(5)
        pdf.image(visual_path, w=180)
    
    pdf.output(filename)
    return filename


# ------------------ Flask Routes ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        try:
            text = extract_article(url)
            summary = generate_summary(text)

            # Generate single-page combined visuals
            visual_path = generate_combined_visuals(summary)

            # Generate PDF
            pdf_path = generate_pdf(summary, visual_path)

            return render_template("result.html",
                                   summary=summary,
                                   pdf=pdf_path,
                                   visual=visual_path)
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("index.html")


@app.route("/static/pdfs/<filename>")
def download_file(filename):
    return send_from_directory("static/pdfs", filename)


# ------------------ Run App ------------------
if __name__ == "__main__":
    # Host 0.0.0.0 is optional for cloud, but fine
    app.run(host="0.0.0.0", port=5000)
