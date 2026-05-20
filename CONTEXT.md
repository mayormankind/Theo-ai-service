# TheoGrader – Project Context

## 📌 Project Title

TheoGrader — Design and Development of an Intelligent Assessment System for Automated Grading of Theoretical Examination Scripts in Nigerian Universities

---

## 🎯 Project Purpose

This project is an **AI-assisted grading system** designed to support lecturers in evaluating **theoretical (essay/short-answer) examination scripts**.

It is important to note:

> ❗ This system is NOT a fully autonomous grading system.  
> It is a **decision-support tool** that assists lecturers, not replaces them.

The system reduces marking stress by:

- Automating repetitive grading tasks
- Providing structured scoring suggestions
- Offering confidence indicators for transparency

---

## 🧠 Core Idea

The system processes uploaded exam scripts and compares student answers against a lecturer-defined marking guide (rubric).

### High-Level Workflow

1. Lecturer uploads:
   - Scanned exam script (image or PDF)
   - Rubric (structured marking scheme)

2. System pipeline:
   - OCR extracts text from script
   - Text is segmented into answers (Q1, Q2, 1(a), etc.)
   - Answers are cleaned and preprocessed
   - Semantic similarity is computed against rubric points
   - Scores are assigned using weighted logic

3. Output:
   - Per-question score
   - Similarity breakdown
   - Confidence score
   - Lecturer can review and override

---

## ⚙️ System Architecture

### Frontend

- Next.js (Fullstack usage)
- Handles UI, file upload, rubric input, and result display

### Backend / API Layer

- Next.js API routes

### AI Service (Core Intelligence)

- Python (FastAPI)
- Handles OCR, segmentation, embeddings, and scoring

### Database

- PostgreSQL (via Prisma ORM)

---

## 🔍 OCR Strategy (Unified Vision Approach)

The system uses a **GPT-4o-mini Vision-based OCR pipeline**:

### Step 1: Document Conversion

- Raw multi-page PDFs are converted page-by-page to image bytes (JPEG format) using `pdf2image`. Images are processed directly.

### Step 2: High-Fidelity Transcription

- Each page is transcribed verbatim using a specialized `gpt-4o-mini` Vision model prompt, which preserves layout structures, handwritings, and spelling errors.

### Step 3: Identity Correction & normalizations

- The Next.js backend parses the output to identify student identities and performs automatic error corrections on matric numbers (e.g. `1FS/2014986` -> `IFS/20/4986`).

### Key Idea:

> Direct Vision Transcription → Verbatim Text → Unified and Robust Processing pipeline without local engine dependencies.

---

## ✂️ Answer Segmentation

The system splits extracted text into structured answers using **regex pattern detection**.

### Supported Patterns:

- Q1, Q2
- Question 1
- 1(a), 2(b)
- 1., 2., etc.

### Output Format:

```json
{
  "Q1": "Answer text...",
  "Q2": "Answer text..."
}

// still figuring the entire system out though - work in progress
```
