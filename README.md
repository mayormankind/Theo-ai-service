# Intelligent Assessment System API

This is a microservice developed for the Final Year Project: "Design and Development of an Intelligent Assessment System for Automated Grading of Theoretical Examination Scripts in Nigerian Universities."

## Core Features
1. **OCR Module**: Extracts text from uploaded images using Tesseract.
2. **Answer Segmentation**: Uses regex to identify and segment individual questions.
3. **Similarity & Grading**: Utilizes `Sentence-BERT` (all-MiniLM-L6-v2) for semantic text embeddings and compares student answers to an expected rubric using Cosine Similarity.
4. **Custom Scoring**: Maps cosine similarity to strict scoring bands (e.g., >= 0.85 gives full marks).

## Prerequisites (Important!)
You **must** install Tesseract OCR on your system for the OCR feature to work. 
- **Windows**: Download the installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Ensure you add Tesseract-OCR to your system PATH!
- **Linux (Ubuntu)**: `sudo apt-get install tesseract-ocr`
- **Mac**: `brew install tesseract`

## Installation
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server
Start the local server using `uvicorn`:
```bash
uvicorn app.main:app --reload
```
The server will run at `http://127.0.0.1:8000`. You can visit `http://127.0.0.1:8000/docs` to see the automated interactive Swagger UI and test endpoints directly from the browser.

## Example Usage

### 1. Health Check
```bash
curl -X GET http://127.0.0.1:8000/
```

### 2. Similarity Check
```bash
curl -X POST http://127.0.0.1:8000/similarity \
     -H "Content-Type: application/json" \
     -d '{
       "student_answer": "The nucleus controls the cell.",
       "rubric": ["Nucleus is the brain of the cell", "Controls cellular activities"]
     }'
```

### 3. Grading Pipeline
Testing the `/grade` endpoint with a file upload using `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/grade" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@student_script.jpg" \
  -F 'rubric_str={"Q1": [{"point": "Explain the concept.", "weight": 2.0}]}'
```
