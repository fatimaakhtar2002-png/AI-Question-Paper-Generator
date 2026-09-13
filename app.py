from flask import Flask, render_template, request, make_response
from huggingface_hub import InferenceClient

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from datetime import datetime

import io
import os
import html


app = Flask(__name__)


# =========================================================
# HUGGING FACE AI SETUP
# =========================================================

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


# =========================================================
# PDF FONT SETUP
# =========================================================

FONT_NAME = "Helvetica"


possible_fonts = [

    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",

    "/usr/share/fonts/dejavu/DejaVuSans.ttf",

    "C:/Windows/Fonts/DejaVuSans.ttf",

    "C:/Windows/Fonts/arial.ttf"

]


for font_path in possible_fonts:

    if os.path.exists(font_path):

        try:

            pdfmetrics.registerFont(
                TTFont("UnicodeFont", font_path)
            )

            FONT_NAME = "UnicodeFont"

            break

        except Exception:

            pass


# =========================================================
# PDF TEXT SAFETY
# =========================================================

def make_pdf_safe(text):

    replacements = {

        "–": "-",
        "—": "-",
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


    return text


# =========================================================
# AI QUESTION GENERATION
# =========================================================

def generate_questions(
    subject,
    topic,
    language,
    difficulty,
    total_questions,
    total_marks
):

    prompt = f"""
You are an expert university question paper generator.

Create a professional academic question paper.

Subject: {subject}
Topic: {topic}
Question Paper Language: {language}
Difficulty: {difficulty}
Total Questions: {total_questions}
Total Marks: {total_marks}

IMPORTANT RULES:

1. Generate EXACTLY {total_questions} questions.
2. Do not generate extra questions.
3. Every question must be relevant to the given subject and topic.
4. Follow the requested difficulty level: {difficulty}.
5. The complete question paper must be written in {language}.
6. Do NOT use English if the selected language is different, except
   where technical terms are normally used.
7. Use the correct native writing system of the selected language.
8. For Hindi, use Devanagari script.
9. For Urdu, use Urdu script.
10. For French, write in French.
11. For Korean, write in Korean Hangul.
12. For Spanish, write in Spanish.
13. For German, write in German.
14. For Japanese, use Japanese writing.
15. For Chinese, use Chinese characters.
16. For Arabic, use Arabic script.
17. If Other is selected, follow the language requested by the user.
18. Divide questions approximately into:
    - 50% MCQs
    - 30% Short Answer
    - 20% Long Answer.
19. Every MCQ must have exactly four options:
    A)
    B)
    C)
    D)
20. Do not repeat questions.
21. Do not include an answer key.
22. Do not use tables.
23. Keep the questions suitable for an academic question paper.
24. Make questions clear and grammatically correct.
25. Make sure the final paper contains EXACTLY {total_questions} questions.
26. Do not add explanations before or after the paper.
27. Return ONLY the question paper.

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
                    "paper generator. Always follow the selected "
                    "question paper language exactly. Generate "
                    "clear academic questions and never add "
                    "unnecessary explanations."
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

        "language": language,

        "difficulty": difficulty,

        "questions": total_questions,

        "total_questions": total_questions,

        "marks": total_marks,

        "date":
            datetime.now().strftime("%d-%m-%Y"),

        "time": "3 Hours",

        "content": ai_text

    }


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# GENERATE QUESTION PAPER
# =========================================================

@app.route(
    "/generate",
    methods=["POST"]
)
def generate():

    subject = request.form.get(
        "subject",
        ""
    )


    topic = request.form.get(
        "topic",
        ""
    )


    language = request.form.get(
        "language",
        "English"
    )


    questions = request.form.get(
        "total_questions",
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

        questions = int(
            questions
        )

    except (ValueError, TypeError):

        questions = 10


    try:

        marks = int(
            marks
        )

    except (ValueError, TypeError):

        marks = 50


    # Safety limits

    if questions < 1:

        questions = 1


    if questions > 100:

        questions = 100


    if marks < 1:

        marks = 1


    if marks > 500:

        marks = 500


    paper = generate_questions(

        subject,

        topic,

        language,

        difficulty,

        questions,

        marks

    )


    return render_template(

        "result.html",

        paper=paper

    )


# =========================================================
# PDF PAGE NUMBER
# =========================================================

def add_page_number(
    canvas,
    doc
):

    canvas.saveState()


    canvas.setFont(
        FONT_NAME,
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


# =========================================================
# DOWNLOAD PDF
# =========================================================

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


    language = request.args.get(
        "language",
        "English"
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


    subject = make_pdf_safe(
        subject
    )


    topic = make_pdf_safe(
        topic
    )


    language = make_pdf_safe(
        language
    )


    difficulty = make_pdf_safe(
        difficulty
    )


    content = make_pdf_safe(
        content
    )


    # =====================================================
    # PDF BUFFER
    # =====================================================

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


    # =====================================================
    # STYLES
    # =====================================================

    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(

        "CustomTitle",

        parent=styles["Title"],

        fontName=FONT_NAME,

        fontSize=18,

        leading=22,

        alignment=TA_CENTER,

        spaceAfter=8,

        textColor=colors.HexColor(
            "#247a9b"
        )

    )


    subtitle_style = ParagraphStyle(

        "Subtitle",

        parent=styles["Normal"],

        fontName=FONT_NAME,

        fontSize=10,

        leading=14,

        alignment=TA_CENTER,

        spaceAfter=15,

        textColor=colors.HexColor(
            "#527987"
        )

    )


    details_style = ParagraphStyle(

        "Details",

        parent=styles["Normal"],

        fontName=FONT_NAME,

        fontSize=10,

        leading=16,

        spaceAfter=2

    )


    section_style = ParagraphStyle(

        "Section",

        parent=styles["Heading2"],

        fontName=FONT_NAME,

        fontSize=13,

        leading=17,

        spaceBefore=12,

        spaceAfter=8,

        textColor=colors.HexColor(
            "#247a9b"
        )

    )


    question_style = ParagraphStyle(

        "Question",

        parent=styles["Normal"],

        fontName=FONT_NAME,

        fontSize=10,

        leading=15,

        spaceAfter=5

    )


    # =====================================================
    # STORY
    # =====================================================

    story = []


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

            f"<b>Language:</b> "
            f"{html.escape(language)}",

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
            f"{html.escape(str(questions))}",

            details_style

        )

    )


    story.append(

        Paragraph(

            f"<b>Total Marks:</b> "
            f"{html.escape(str(marks))}",

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


    story.append(

        Paragraph(

            "------------------------------------------------------------",

            question_style

        )

    )


    story.append(
        Spacer(1, 8)
    )


    # =====================================================
    # AI GENERATED CONTENT
    # =====================================================

    lines = content.splitlines()


    for line in lines:

        line = line.strip()


        if not line:

            story.append(
                Spacer(1, 5)
            )

            continue


        line = line.replace(
            "**",
            ""
        )


        upper_line = line.upper()


        if (

            "SECTION A" in upper_line

            or

            "SECTION B" in upper_line

            or

            "SECTION C" in upper_line

        ):

            story.append(

                Paragraph(

                    html.escape(line),

                    section_style

                )

            )

            continue


        story.append(

            Paragraph(

                html.escape(line),

                question_style

            )

        )


    # =====================================================
    # BUILD PDF
    # =====================================================

    pdf.build(

        story,

        onFirstPage=add_page_number,

        onLaterPages=add_page_number

    )


    buffer.seek(0)


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


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )