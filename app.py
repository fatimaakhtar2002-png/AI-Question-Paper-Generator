from flask import Flask, render_template, request, send_file
from huggingface_hub import InferenceClient
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from datetime import datetime
import os
import io
import re


app = Flask(__name__)

MODEL = "openai/gpt-oss-120b"


# --------------------------------------------------
# HUGGING FACE CLIENT
# --------------------------------------------------

def get_hf_client():
    token = os.environ.get("HF_TOKEN")

    if not token:
        return None

    return InferenceClient(
        model=MODEL,
        token=token
    )


# --------------------------------------------------
# GENERATED PAPER STORAGE
# --------------------------------------------------

generated_paper = {
    "subject": "",
    "topic": "",
    "language": "English",
    "difficulty": "Medium",
    "total_questions": 10,
    "marks": 50,
    "content": "",
    "date": "",
    "time": ""
}


# --------------------------------------------------
# LANGUAGE INSTRUCTIONS
# --------------------------------------------------

def language_instruction(language):

    instructions = {
        "English": "Write everything in English.",
        "Hindi": "Write everything in Hindi using Devanagari script.",
        "Urdu": "Write everything in Urdu using Urdu script.",
        "French": "Write everything in French.",
        "Korean": "Write everything in Korean.",
        "Spanish": "Write everything in Spanish.",
        "German": "Write everything in German.",
        "Japanese": "Write everything in Japanese.",
        "Chinese": "Write everything in Chinese.",
        "Arabic": "Write everything in Arabic.",
        "Other": "Write everything in the language selected by the user."
    }

    return instructions.get(
        language,
        "Write everything in the selected language."
    )


# --------------------------------------------------
# QUESTION DISTRIBUTION
# --------------------------------------------------

def get_distribution(total_questions):

    # For 1 question
    if total_questions == 1:
        return 1, 0, 0

    # For 2 questions
    if total_questions == 2:
        return 1, 1, 0

    # For 3 or more questions
    mcq_count = round(total_questions * 0.5)
    short_count = round(total_questions * 0.3)
    long_count = total_questions - mcq_count - short_count

    # Safety correction
    if mcq_count < 1:
        mcq_count = 1

    if short_count < 1:
        short_count = 1

    if long_count < 1:
        long_count = 1

    # Final correction if total goes above requested number
    while mcq_count + short_count + long_count > total_questions:

        if mcq_count > 1:
            mcq_count -= 1

        elif short_count > 1:
            short_count -= 1

        elif long_count > 1:
            long_count -= 1

        else:
            break

    # Final correction if total is below requested number
    while mcq_count + short_count + long_count < total_questions:
        long_count += 1

    return mcq_count, short_count, long_count


# --------------------------------------------------
# FORMAT AI OUTPUT
# --------------------------------------------------

def clean_output(text):

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove markdown code blocks
    text = re.sub(
        r"```(?:text|markdown)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace("```", "")

    # Remove unnecessary markdown headings
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # --------------------------------------------------
    # INSERT NEW LINES BEFORE SECTIONS
    # --------------------------------------------------

    text = re.sub(
        r"\s*(SECTION\s+A\s*[-–—:]\s*MCQs?)",
        r"\n\nSECTION A - MCQs\n\n",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*(SECTION\s+B\s*[-–—:]\s*SHORT\s+ANSWER)",
        r"\n\nSECTION B - SHORT ANSWER\n\n",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*(SECTION\s+C\s*[-–—:]\s*LONG\s+ANSWER)",
        r"\n\nSECTION C - LONG ANSWER\n\n",
        text,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------
    # INSERT NEW LINE BEFORE QUESTION NUMBERS
    # --------------------------------------------------

    # Handles:
    # 1.
    # 2.
    # 10.
    #
    # Also handles:
    # 1)
    # 2)
    # 10)

    text = re.sub(
        r"\s+(\d{1,3}\s*[\.\)])\s+",
        r"\n\n\1 ",
        text
    )

    # --------------------------------------------------
    # INSERT NEW LINE BEFORE MCQ OPTIONS
    # --------------------------------------------------

    text = re.sub(
        r"\s+([A-Da-d]\s*[\)\.\:])\s+",
        r"\n\1 ",
        text
    )

    # --------------------------------------------------
    # REMOVE EXCESSIVE SPACES
    # --------------------------------------------------

    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue

        # Remove extra spaces
        line = re.sub(r"[ \t]+", " ", line)

        lines.append(line)

    # Remove excessive blank lines
    cleaned_lines = []

    blank_count = 0

    for line in lines:

        if line == "":
            blank_count += 1

            if blank_count <= 2:
                cleaned_lines.append("")

        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# --------------------------------------------------
# GENERATE QUESTIONS USING HUGGING FACE
# --------------------------------------------------

def generate_questions(
    subject,
    topic,
    language,
    difficulty,
    total_questions,
    total_marks
):

    client = get_hf_client()

    if client is None:
        return (
            "Hugging Face token is not configured. "
            "Please set the HF_TOKEN environment variable."
        )

    # Get section distribution
    mcq_count, short_count, long_count = get_distribution(
        total_questions
    )

    lang_rule = language_instruction(language)

    # --------------------------------------------------
    # AI PROMPT
    # --------------------------------------------------

    prompt = f"""
You are a professional university examination question paper setter.

Create a complete question paper based on the information below.

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

MCQs: {mcq_count}
SHORT ANSWER: {short_count}
LONG ANSWER: {long_count}


IMPORTANT RULES:

1. Generate EXACTLY {total_questions} questions.
2. Number the questions continuously from 1 to {total_questions}.
3. Do not skip any question number.
4. Do not repeat questions.
5. Every question must be related to the subject and topic.
6. Follow the requested difficulty level.
7. {lang_rule}
8. Do NOT provide answers.
9. Do NOT provide an answer key.
10. Do NOT provide explanations.
11. Do NOT use markdown tables.
12. Do NOT use code blocks.
13. Keep every question clearly separated.
14. Put a blank line after every question.
15. Every MCQ must have exactly four options.
16. Use A), B), C), D) for MCQ options.
17. Each MCQ option must be written on a separate line.
18. Each question number must start on a new line.
19. Section headings must start on a new line.
20. The total marks of all questions must equal exactly {total_marks}.
21. Do not add any introduction before SECTION A.
22. Do not add any conclusion after the questions.


FORMAT EXACTLY LIKE THIS:


SECTION A - MCQs

1. (marks) Question text

A) Option
B) Option
C) Option
D) Option


2. (marks) Question text

A) Option
B) Option
C) Option
D) Option


SECTION B - SHORT ANSWER

Continue question numbering.

Example:

6. (marks) Question text


7. (marks) Question text


SECTION C - LONG ANSWER

Continue question numbering.

Example:

9. (marks) Question text


10. (marks) Question text


MARKS:

The sum of all question marks must be exactly {total_marks}.


VERY IMPORTANT:

Start directly with:

SECTION A - MCQs

Do not write anything before it.
Do not provide answers.
"""


    # --------------------------------------------------
    # CALL HUGGING FACE
    # --------------------------------------------------

    try:

        response = client.chat_completion(

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert university question "
                        "paper generator. Follow the requested "
                        "question count, language, difficulty, "
                        "section structure and total marks exactly."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            max_tokens=6000,

            temperature=0.7
        )

        content = response.choices[0].message.content

        # Format output
        content = clean_output(content)

        return content

    except Exception as e:

        return (
            "Unable to generate the question paper.\n\n"
            "Error: "
            + str(e)
        )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------------------------
# GENERATE PAGE
# --------------------------------------------------

@app.route("/generate", methods=["POST"])
def generate():

    global generated_paper

    subject = request.form.get(
        "subject",
        ""
    ).strip()

    topic = request.form.get(
        "topic",
        ""
    ).strip()

    language = request.form.get(
        "language",
        "English"
    ).strip()

    difficulty = request.form.get(
        "difficulty",
        "Medium"
    ).strip()


    # Total Questions
    try:

        total_questions = int(
            request.form.get(
                "total_questions",
                "10"
            )
        )

    except ValueError:

        total_questions = 10


    # Total Marks
    try:

        total_marks = int(
            request.form.get(
                "total_marks",
                "50"
            )
        )

    except ValueError:

        total_marks = 50


    # Validation
    total_questions = max(
        1,
        min(total_questions, 100)
    )

    total_marks = max(
        1,
        min(total_marks, 500)
    )


    # Generate AI questions
    content = generate_questions(

        subject=subject,

        topic=topic,

        language=language,

        difficulty=difficulty,

        total_questions=total_questions,

        total_marks=total_marks
    )


    # Current date and time
    now = datetime.now()


    generated_paper = {

        "subject": subject,

        "topic": topic,

        "language": language,

        "difficulty": difficulty,

        "total_questions": total_questions,

        "marks": total_marks,

        "content": content,

        "date": now.strftime(
            "%d-%m-%Y"
        ),

        "time": now.strftime(
            "%I:%M %p"
        )
    }


    return render_template(
        "result.html",
        paper=generated_paper
    )


# --------------------------------------------------
# DOWNLOAD PDF
# --------------------------------------------------

@app.route("/download")
def download():

    subject = generated_paper.get(
        "subject",
        "Question Paper"
    )

    topic = generated_paper.get(
        "topic",
        ""
    )

    language = generated_paper.get(
        "language",
        "English"
    )

    difficulty = generated_paper.get(
        "difficulty",
        "Medium"
    )

    total_questions = generated_paper.get(
        "total_questions",
        10
    )

    total_marks = generated_paper.get(
        "marks",
        50
    )

    content = generated_paper.get(
        "content",
        ""
    )


    # Create PDF in memory
    buffer = io.BytesIO()


    document = SimpleDocTemplate(

        buffer,

        pagesize=A4,

        rightMargin=45,

        leftMargin=45,

        topMargin=45,

        bottomMargin=45
    )


    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(

        "TitleStyle",

        parent=styles["Title"],

        fontSize=18,

        leading=22,

        alignment=TA_CENTER,

        textColor=colors.HexColor(
            "#168aad"
        ),

        spaceAfter=20
    )


    info_style = ParagraphStyle(

        "InfoStyle",

        parent=styles["Normal"],

        fontSize=10.5,

        leading=15,

        spaceAfter=5
    )


    section_style = ParagraphStyle(

        "SectionStyle",

        parent=styles["Heading2"],

        fontSize=13,

        leading=17,

        textColor=colors.HexColor(
            "#168aad"
        ),

        spaceBefore=18,

        spaceAfter=12
    )


    question_style = ParagraphStyle(

        "QuestionStyle",

        parent=styles["Normal"],

        fontSize=10.5,

        leading=16,

        spaceAfter=10
    )


    option_style = ParagraphStyle(

        "OptionStyle",

        parent=styles["Normal"],

        fontSize=10.5,

        leading=15,

        leftIndent=18,

        spaceAfter=4
    )


    story = []


    # Title
    story.append(
        Paragraph(
            "AI Generated Question Paper",
            title_style
        )
    )


    # Paper information
    story.append(
        Paragraph(
            f"<b>Subject:</b> {subject}",
            info_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Topic:</b> {topic}",
            info_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Language:</b> {language}",
            info_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Difficulty:</b> {difficulty}",
            info_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Total Questions:</b> {total_questions}",
            info_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Total Marks:</b> {total_marks}",
            info_style
        )
    )


    story.append(
        Spacer(1, 15)
    )


    # --------------------------------------------------
    # ADD GENERATED CONTENT TO PDF
    # --------------------------------------------------

    for line in content.split("\n"):

        line = line.strip()

        if not line:

            story.append(
                Spacer(1, 8)
            )

            continue


        # Escape HTML characters
        safe_line = (
            line
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )


        upper_line = line.upper()


        # Section headings
        if (
            "SECTION A" in upper_line
            or "SECTION B" in upper_line
            or "SECTION C" in upper_line
        ):

            story.append(
                Paragraph(
                    f"<b>{safe_line}</b>",
                    section_style
                )
            )


        # MCQ options
        elif re.match(
            r"^[A-Da-d][\)\.\:]\s*",
            line
        ):

            story.append(
                Paragraph(
                    safe_line,
                    option_style
                )
            )


        # Questions / normal text
        else:

            story.append(
                Paragraph(
                    safe_line,
                    question_style
                )
            )


    # Build PDF
    document.build(story)


    buffer.seek(0)


    return send_file(

        buffer,

        as_attachment=True,

        download_name="AI_Question_Paper.pdf",

        mimetype="application/pdf"
    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )