import os
import re
from datetime import datetime

from flask import Flask, render_template, request, send_file
from huggingface_hub import InferenceClient

from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from xml.sax.saxutils import escape


app = Flask(__name__)

# Working Hugging Face model
MODEL = "openai/gpt-oss-120b"

generated_paper = None


# =========================================================
# HUGGING FACE AI CLIENT
# =========================================================

def get_ai_client():
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Please set your Hugging Face token."
        )

    return InferenceClient(token=token)


# =========================================================
# LANGUAGE INSTRUCTIONS
# =========================================================

LANGUAGE_INSTRUCTIONS = {
    "English": "Generate all questions and instructions in English.",
    "Hindi": "Generate all questions and instructions in Hindi using Devanagari script.",
    "Urdu": "Generate all questions and instructions in Urdu using Urdu/Arabic script.",
    "French": "Generate all questions and instructions in French.",
    "Korean": "Generate all questions and instructions in Korean.",
    "Spanish": "Generate all questions and instructions in Spanish.",
    "German": "Generate all questions and instructions in German.",
    "Japanese": "Generate all questions and instructions in Japanese.",
    "Chinese": "Generate all questions and instructions in Simplified Chinese.",
    "Arabic": "Generate all questions and instructions in Arabic.",
    "Other": "Generate the questions in the requested language."
}


# =========================================================
# QUESTION DISTRIBUTION
# =========================================================

def calculate_distribution(total_questions):
    if total_questions <= 0:
        return 0, 0, 0

    if total_questions == 1:
        return 1, 0, 0

    if total_questions == 2:
        return 1, 1, 0

    mcq = round(total_questions * 0.50)
    short = round(total_questions * 0.30)
    long = total_questions - mcq - short

    if long < 1:
        long = 1

        if mcq > short:
            mcq -= 1
        else:
            short -= 1

    return mcq, short, long


# =========================================================
# AI QUESTION GENERATION
# =========================================================

def generate_questions(
    subject,
    topic,
    language,
    total_questions,
    difficulty,
    total_marks
):
    client = get_ai_client()

    mcq_count, short_count, long_count = calculate_distribution(
        total_questions
    )

    language_instruction = LANGUAGE_INSTRUCTIONS.get(
        language,
        LANGUAGE_INSTRUCTIONS["Other"]
    )

    prompt = f"""
You are an expert university examination question paper generator.

Create a complete question paper.

SUBJECT:
{subject}

TOPIC:
{topic}

LANGUAGE:
{language}

DIFFICULTY:
{difficulty}

TOTAL QUESTIONS:
{total_questions}

TOTAL MARKS:
{total_marks}

QUESTION DISTRIBUTION:

MCQ:
{mcq_count}

SHORT ANSWER:
{short_count}

LONG ANSWER:
{long_count}

LANGUAGE INSTRUCTION:
{language_instruction}

IMPORTANT RULES:

1. Generate EXACTLY {total_questions} questions.
2. Do not generate fewer questions.
3. Do not generate more questions.
4. Total marks must be exactly {total_marks}.
5. Questions must be relevant to the subject and topic.
6. Maintain the requested difficulty level.
7. Do not provide answers.
8. Do not provide solutions.
9. Do not provide an answer key.
10. MCQs must contain exactly four options.
11. MCQ options must be A, B, C and D.
12. Use clear numbering.
13. Keep proper spacing.
14. Do not use a Markdown table.
15. Do not add unnecessary explanation.
16. Do not write anything before SECTION A.
17. Do not write anything after the final question.

Use exactly this structure:

SECTION A - MULTIPLE CHOICE QUESTIONS

1. Question text
A. Option
B. Option
C. Option
D. Option
Marks: X

SECTION B - SHORT ANSWER QUESTIONS

2. Question text
Marks: X

SECTION C - LONG ANSWER QUESTIONS

3. Question text
Marks: X

The sum of all question marks must be exactly {total_marks}.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert academic question paper "
                        "generator. Follow the requested structure exactly."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=12000
        )
    except Exception as e:
        raise RuntimeError(f"Hugging Face AI error: {str(e)}")

    if not response.choices:
        raise RuntimeError("AI did not return a response.")

    message = response.choices[0].message
    content = getattr(message, "content", None)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    parts.append(str(value))
            else:
                value = getattr(item, "text", None) or getattr(item, "content", None)
                if value:
                    parts.append(str(value))
        content = "\n".join(parts)

    if not content:
        raise RuntimeError(
            "AI returned empty content. Please try generating the paper again."
        )

    return clean_output(str(content))


# =========================================================
# CLEAN AI OUTPUT
# =========================================================

def clean_output(text):
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"```(?:text|markdown)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace("```", "")
    text = text.strip()

    text = re.sub(
        r"\s*SECTION\s*A\s*[-:]?\s*",
        "\n\nSECTION A - MULTIPLE CHOICE QUESTIONS\n\n",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*SECTION\s*B\s*[-:]?\s*",
        "\n\nSECTION B - SHORT ANSWER QUESTIONS\n\n",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*SECTION\s*C\s*[-:]?\s*",
        "\n\nSECTION C - LONG ANSWER QUESTIONS\n\n",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s+(\d+)\.\s+",
        r"\n\1. ",
        text
    )

    text = re.sub(
        r"\s+([A-D])\.\s+",
        r"\n\1. ",
        text
    )

    text = re.sub(
        r"\s+(Marks?\s*:\s*)",
        r"\n\1",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\n{4,}",
        "\n\n\n",
        text
    )

    return text.strip()


# =========================================================
# PDF FONTS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(BASE_DIR, "fonts")

FONT_FILES = {
    "NotoSans": "NotoSans-Regular.ttf",
    "Devanagari": "NotoSansDevanagari-Regular.ttf",
    "Arabic": "NotoNaskhArabic-Regular.ttf",
    "Japanese": "NotoSansJP-Regular.ttf",
    "Korean": "NotoSansKR-Regular.ttf",
    "Chinese": "NotoSansSC-Regular.ttf"
}


def register_fonts():
    registered = {}

    for font_name, filename in FONT_FILES.items():
        path = os.path.join(FONT_DIR, filename)

        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, path))
                registered[font_name] = True
            except Exception:
                registered[font_name] = False
        else:
            registered[font_name] = False

    return registered


FONT_STATUS = register_fonts()


# =========================================================
# SELECT PDF FONT
# =========================================================

def get_pdf_font(language):
    if language == "Hindi" and FONT_STATUS.get("Devanagari"):
        return "Devanagari"

    if language in ["Urdu", "Arabic"] and FONT_STATUS.get("Arabic"):
        return "Arabic"

    if language == "Korean" and FONT_STATUS.get("Korean"):
        return "Korean"

    if language == "Japanese" and FONT_STATUS.get("Japanese"):
        return "Japanese"

    if language == "Chinese" and FONT_STATUS.get("Chinese"):
        return "Chinese"

    if FONT_STATUS.get("NotoSans"):
        return "NotoSans"

    return "Helvetica"


# =========================================================
# GENERATE PDF
# =========================================================

def make_mixed_script_markup(text, script_font, latin_font="Helvetica"):
    """Return ReportLab markup that explicitly uses Helvetica for Latin text
    and the selected Unicode font for the requested script.
    This prevents English labels/option letters from disappearing when a
    script-specific font is used for Hindi/Arabic/CJK text.
    """
    safe = escape(text)

    # Latin letters, digits and common exam punctuation. Keep runs together
    # so things such as "SECTION A", "1.", "A.", "Marks: 2" render reliably.
    pattern = r"[A-Za-z0-9][A-Za-z0-9 .,:;!?()/%+\-*=\'\"]*"

    return re.sub(
        pattern,
        lambda m: f'<font name="{latin_font}">{m.group(0)}</font>',
        safe
    )


def generate_pdf(paper):
    language = paper["language"]
    font_name = get_pdf_font(language)

    filepath = os.path.join(
        BASE_DIR,
        "generated_question_paper.pdf"
    )

    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AI Question Paper Generator"
    )

    styles = getSampleStyleSheet()

    # Use the Latin font for the paper title and metadata so English
    # labels remain visible even when the selected language uses a
    # script-specific font.
    latin_font = "Helvetica"

    title_style = ParagraphStyle(
        "PaperTitle",
        parent=styles["Title"],
        fontName=latin_font,
        fontSize=17,
        leading=22,
        alignment=TA_LEFT,
        spaceAfter=10
    )

    info_style = ParagraphStyle(
        "Info",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10.5,
        leading=16,
        spaceAfter=3
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName=latin_font,
        fontSize=13,
        leading=18,
        spaceBefore=12,
        spaceAfter=8
    )

    question_style = ParagraphStyle(
        "Question",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10.5,
        leading=17,
        spaceAfter=5
    )

    if language in ["Urdu", "Arabic"]:
        title_style.alignment = TA_RIGHT
        info_style.alignment = TA_RIGHT
        section_style.alignment = TA_RIGHT
        question_style.alignment = TA_RIGHT

    story = []

    story.append(
        Paragraph(
            "AI QUESTION PAPER GENERATOR",
            title_style
        )
    )

    metadata = [
        ("Subject", str(paper["subject"])),
        ("Topic", str(paper["topic"])),
        ("Language", str(paper["language"])),
        ("Difficulty", str(paper["difficulty"])),
        ("Total Questions", str(paper["total_questions"])),
        ("Total Marks", str(paper["total_marks"])),
    ]

    for label, value in metadata:
        if font_name != latin_font:
            metadata_markup = (
                f'<font name="{latin_font}"><b>{escape(label)}:</b></font> '
                f'{make_mixed_script_markup(value, font_name, latin_font)}'
            )
        else:
            metadata_markup = f'<b>{escape(label)}:</b> {escape(value)}'

        story.append(Paragraph(metadata_markup, info_style))

    story.append(Spacer(1, 8))

    for line in paper["generated_content"].split("\n"):
        clean_line = line.strip()

        if not clean_line:
            story.append(Spacer(1, 5))
            continue

        # Explicitly switch fonts inside mixed-language lines.
        # Helvetica is used for Latin labels/numbers because it is built into
        # ReportLab and therefore cannot lose A-Z, digits, punctuation, etc.
        if font_name != latin_font:
            safe_line = make_mixed_script_markup(
                clean_line,
                script_font=font_name,
                latin_font=latin_font
            )
        else:
            safe_line = escape(clean_line)

        if re.match(
            r"^SECTION\s+[ABC]",
            clean_line,
            flags=re.IGNORECASE
        ):
            story.append(
                Paragraph(
                    safe_line,
                    section_style
                )
            )
        else:
            story.append(
                Paragraph(
                    safe_line,
                    question_style
                )
            )

    doc.build(story)

    return filepath


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")


# =========================================================
# GENERATE QUESTION PAPER
# =========================================================

@app.route("/generate", methods=["POST"])
def generate():
    global generated_paper

    try:
        subject = request.form.get("subject", "").strip()
        topic = request.form.get("topic", "").strip()
        language = request.form.get("language", "English").strip()
        difficulty = request.form.get("difficulty", "Medium").strip()

        total_questions_raw = request.form.get(
            "total_questions",
            "10"
        ).strip()

        total_marks_raw = request.form.get(
            "total_marks",
            "50"
        ).strip()

        if not subject:
            raise ValueError("Please enter the subject.")

        if not topic:
            raise ValueError("Please enter the topic.")

        try:
            total_questions = int(total_questions_raw)
        except ValueError:
            raise ValueError("Total Questions must be a number.")

        try:
            total_marks = float(total_marks_raw)
        except ValueError:
            raise ValueError("Total Marks must be a number.")

        if total_questions < 1:
            raise ValueError("Total Questions must be at least 1.")

        if total_questions > 100:
            raise ValueError("Maximum 100 questions are allowed.")

        if total_marks <= 0:
            raise ValueError("Total Marks must be greater than 0.")

        if total_marks.is_integer():
            total_marks = int(total_marks)

        ai_output = generate_questions(
            subject=subject,
            topic=topic,
            language=language,
            total_questions=total_questions,
            difficulty=difficulty,
            total_marks=total_marks
        )

        generated_paper = {
            "subject": subject,
            "topic": topic,
            "language": language,
            "difficulty": difficulty,
            "total_questions": total_questions,
            "total_marks": total_marks,
            "marks": total_marks,
            "generated_content": ai_output,
            "content": ai_output,
            "date": datetime.now().strftime("%d-%m-%Y"),
            "time": datetime.now().strftime("%I:%M %p")
        }

        return render_template(
            "result.html",
            paper=generated_paper,
            generated_content=ai_output
        )

    except Exception as e:
        error_message = str(e)

        return render_template(
            "result.html",
            paper={
                "subject": request.form.get("subject", ""),
                "topic": request.form.get("topic", ""),
                "language": request.form.get("language", "English"),
                "difficulty": request.form.get("difficulty", "Medium"),
                "total_questions": request.form.get("total_questions", ""),
                "total_marks": request.form.get("total_marks", ""),
                "marks": request.form.get("total_marks", ""),
                "generated_content": "",
                "content": "",
                "date": datetime.now().strftime("%d-%m-%Y"),
                "time": datetime.now().strftime("%I:%M %p")
            },
            generated_content="",
            error=error_message
        )


# =========================================================
# DOWNLOAD PDF
# =========================================================

@app.route("/download")
def download():
    global generated_paper

    if not generated_paper:
        return (
            "No question paper available. Please generate a paper first.",
            400
        )

    try:
        filepath = generate_pdf(generated_paper)

        return send_file(
            filepath,
            as_attachment=True,
            download_name="AI_Question_Paper.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        return (
            f"Unable to generate PDF: {str(e)}",
            500
        )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
