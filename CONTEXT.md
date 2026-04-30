# Intelligent Assessment System – Project Context

## 📌 Project Title

Design and Development of an Intelligent Assessment System for Automated Grading of Theoretical Examination Scripts in Nigerian Universities

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

## 🔍 OCR Strategy (Hybrid Approach)

The system uses a **hybrid OCR pipeline**:

### Step 1: Primary OCR

- Tesseract (fast, offline, free)

### Step 2: Quality Evaluation

- Checks:
  - Text length
  - Noise ratio
  - Readability

### Step 3: Fallback OCR (if needed)

- GPT-4 Vision (or equivalent AI model)

### Step 4: Output Selection

- Use best available result
- Track method used:
  - `tesseract`
  - `gpt4`
  - `hybrid`

### Key Idea:

> Cheap first → Smart fallback → Reliable output

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
