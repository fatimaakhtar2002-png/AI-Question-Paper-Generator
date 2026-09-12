from flask import Flask, render_template, request, make_response
from huggingface_hub import InferenceClient
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from datetime import datetime
import io
import os
import html


app = Flask(__name__)


# -----------------------------
# Hugging Face AI Setup
# -----------------------------

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN nahi mila. Pehle PowerShell mein token set karo."
    )


client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)


MODEL = "openai/gpt-oss-120b"


# -----------------------------
# Make Text PDF Safe
# -----------------------------

def make_pdf_safe(text):

    # Replace common Unicode punctuation
    replacements = {

        "–": "-",
        "—": "-",
        "-": "-",
        "−": "-",

        "“": '"',
        "”": '"',

        "‘": "'",
        "’": "'",

        "…": "...",

        "•": "-",

        "→": "->",
        "←": "<-",

        "×": "x",
        "÷": "/",

        "\u00a0": " "
    }


    for old, new in replacements.items():

        text = text.replace(old, new)


    # Remove any remaining unsupported Unicode characters
    text = text.encode(
        "ascii",
        "ignore"
    ).decode(
        "ascii"
    )


    return text


# -----------------------------
# AI Question Generation
# -----------------------------

def generate_questions(
    subject,
    topic,
    difficulty,
    total_questions,
    total_marks
):

    prompt = f"""
You are an expert university question paper generator.

Create a professional academic question paper.

Subject: {subject}
Topic: {topic}
Difficulty: {difficulty}
Total Questions: {total_questions}
Total Marks: {total_marks}

IMPORTANT RULES:

1. Generate EXACTLY {total_questions} questions.
2. Do not generate extra questions.
3. All questions must be relevant to the given subject and topic.
4. Follow the requested difficulty level: {difficulty}.
5. Divide questions approximately into:
   - 50% MCQs
   - 30% Short Answer
   - 20% Long Answer.
6. Every MCQ must have exactly four options:
   A)
   B)
   C)
   D)
7. Do not repeat questions.
8. Do not include an answer key.
9. Do not use tables.
10. Keep the questions suitable for an academic question paper.
11. Make the questions clear and grammatically correct.
12. Make sure the final paper contains EXACTLY {total_questions} questions.
13. Write the entire question paper in ENGLISH ONLY.
14. Do not use Hindi, Devanagari, emojis, or any non-English characters.
15. Use only standard English letters, numbers and punctuation.
16. Do not use special Unicode dashes or quotation marks.
17. Use a normal hyphen (-) when a hyphen is required.

Use this format:

SECTION A - MCQs

1. Question
A) Option
B) Option
C) Option
D) Option

SECTION B - SHORT ANSWER

...

SECTION C - LONG ANSWER

...

Return ONLY the question paper.
"""


    response = client.chat.completions.create(

        model=MODEL,

        messages=[

            {
                "role": "system",
                "content": (
                    "You are an expert educational question "
                    "paper generator. Generate English-only "
                    "academic content using standard ASCII "
                    "characters."
                )
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        max_tokens=6000
    )


    ai_text = response.choices[0].message.content


    return {

        "subject": subject,

        "topic": topic,

        "difficulty": difficulty,

        "questions": total_questions,

        "marks": total_marks,

        "date":
            datetime.now().strftime("%d-%m-%Y"),

        "time": "3 Hours",

        "content": ai_text
    }


# -----------------------------
# Home Page
# -----------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -----------------------------
# Generate Question Paper
# -----------------------------

@app.route("/generate", methods=["POST"])
def generate():

    subject = request.form.get(
        "subject",
        ""
    )

    topic = request.form.get(
        "topic",
        ""
    )

    questions = request.form.get(
        "questions",
        "10"
    )

    difficulty = request.form.get(
        "difficulty",
        "Medium"
    )

    marks = request.form.get(
        "marks",
        "50"
    )


    try:

        questions = int(questions)

    except ValueError:

        questions = 10


    try:

        marks = int(marks)

    except ValueError:

        marks = 50


    paper = generate_questions(

        subject,

        topic,

        difficulty,

        questions,

        marks
    )


    return render_template(

        "result.html",

        paper=paper
    )


# -----------------------------
# PDF Page Number
# -----------------------------

def add_page_number(canvas, doc):

    canvas.saveState()


    canvas.setFont(
        "Helvetica",
        8
    )


    canvas.setFillColor(
        colors.grey
    )


    canvas.drawCentredString(

        A4[0] / 2,

        20,

        f"Page {doc.page}"
    )


    canvas.restoreState()


# -----------------------------
# Download Professional PDF
# -----------------------------

@app.route("/download")
def download():

    subject = request.args.get(
        "subject",
        ""
    )

    topic = request.args.get(
        "topic",
        ""
    )

    questions = request.args.get(
        "questions",
        "10"
    )

    difficulty = request.args.get(
        "difficulty",
        "Medium"
    )

    marks = request.args.get(
        "marks",
        "50"
    )

    content = request.args.get(
        "content",
        ""
    )


    # Make all PDF text safe
    subject = make_pdf_safe(subject)

    topic = make_pdf_safe(topic)

    difficulty = make_pdf_safe(difficulty)

    content = make_pdf_safe(content)


    # -------------------------
    # PDF Buffer
    # -------------------------

    buffer = io.BytesIO()


    pdf = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=45,

        leftMargin=45,

        topMargin=45,

        bottomMargin=40,

        title="AI Question Paper Generator"
    )


    # -------------------------
    # Styles
    # -------------------------

    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(

        "CustomTitle",

        parent=styles["Title"],

        fontName="Helvetica-Bold",

        fontSize=18,

        leading=22,

        alignment=TA_CENTER,

        spaceAfter=8,

        textColor=colors.HexColor("#c45a78")
    )


    subtitle_style = ParagraphStyle(

        "Subtitle",

        parent=styles["Normal"],

        fontName="Helvetica",

        fontSize=10,

        leading=14,

        alignment=TA_CENTER,

        spaceAfter=15,

        textColor=colors.HexColor("#70404e")
    )


    details_style = ParagraphStyle(

        "Details",

        parent=styles["Normal"],

        fontName="Helvetica",

        fontSize=10,

        leading=16,

        spaceAfter=2
    )


    section_style = ParagraphStyle(

        "Section",

        parent=styles["Heading2"],

        fontName="Helvetica-Bold",

        fontSize=13,

        leading=17,

        spaceBefore=12,

        spaceAfter=8,

        textColor=colors.HexColor("#c45a78")
    )


    question_style = ParagraphStyle(

        "Question",

        parent=styles["Normal"],

        fontName="Helvetica",

        fontSize=10,

        leading=15,

        spaceAfter=5
    )


    # -------------------------
    # Story
    # -------------------------

    story = []


    # Title

    story.append(

        Paragraph(

            "AI QUESTION PAPER GENERATOR",

            title_style
        )
    )


    story.append(

        Paragraph(

            "AI Generated Academic Question Paper",

            subtitle_style
        )
    )


    # -------------------------
    # Paper Details
    # -------------------------

    story.append(

        Paragraph(

            f"<b>Subject:</b> "
            f"{html.escape(subject)}",

            details_style
        )
    )


    story.append(

        Paragraph(

            f"<b>Topic:</b> "
            f"{html.escape(topic)}",

            details_style
        )
    )


    story.append(

        Paragraph(

            f"<b>Difficulty Level:</b> "
            f"{html.escape(difficulty)}",

            details_style
        )
    )


    story.append(

        Paragraph(

            f"<b>Total Questions:</b> "
            f"{questions}",

            details_style
        )
    )


    story.append(

        Paragraph(

            f"<b>Total Marks:</b> "
            f"{marks}",

            details_style
        )
    )


    story.append(

        Paragraph(

            "<b>Time:</b> 3 Hours",

            details_style
        )
    )


    story.append(

        Paragraph(

            f"<b>Date:</b> "
            f"{datetime.now().strftime('%d-%m-%Y')}",

            details_style
        )
    )


    story.append(
        Spacer(1, 15)
    )


    # -------------------------
    # Separator
    # -------------------------

    story.append(

        Paragraph(

            "------------------------------------------------------------",

            question_style
        )
    )


    story.append(
        Spacer(1, 8)
    )


    # -------------------------
    # AI Generated Content
    # -------------------------

    lines = content.splitlines()


    for line in lines:

        line = line.strip()


        if not line:

            story.append(
                Spacer(1, 5)
            )

            continue


        upper_line = line.upper()


        # Remove markdown stars from section headings

        line = line.replace(
            "**",
            ""
        )


        # Section headings

        if (
            "SECTION A" in upper_line
            or "SECTION B" in upper_line
            or "SECTION C" in upper_line
        ):

            story.append(

                Paragraph(

                    html.escape(line),

                    section_style
                )
            )

            continue


        # Normal question / option

        story.append(

            Paragraph(

                html.escape(line),

                question_style
            )
        )


    # -------------------------
    # Build PDF
    # -------------------------

    pdf.build(

        story,

        onFirstPage=add_page_number,

        onLaterPages=add_page_number
    )


    buffer.seek(0)


    # -------------------------
    # PDF Response
    # -------------------------

    response = make_response(
        buffer.read()
    )


    response.headers[
        "Content-Type"
    ] = "application/pdf"


    response.headers[
        "Content-Disposition"
    ] = (
        "attachment; "
        "filename=AI_Question_Paper.pdf"
    )


    return response


# -----------------------------
# Run Application
# -----------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )