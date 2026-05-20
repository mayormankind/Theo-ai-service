# TheoGrader AI Service API

This is a microservice developed for the Final Year Project: "Design and Development of an Intelligent Assessment System for Automated Grading of Theoretical Examination Scripts in Nigerian Universities."

## Core Features
1. **OCR Module**: Extracts text verbatim from uploaded images and multi-page PDFs using **GPT-4o-mini Vision** via the OpenAI API (with `pdf2image` conversion support), removing the need for error-prone local Tesseract installations.
2. **Answer Segmentation**: Uses regex to identify and segment individual questions (e.g. Q1, 2(a)) with an intelligent `gpt-4o-mini` JSON fallback if regular expression parsing fails.
3. **Similarity & Grading**: Utilizes OpenAI's **`text-embedding-3-small`** for high-fidelity 1536-dimension semantic text embeddings, comparing student answers to an expected rubric using Cosine Similarity.
4. **Custom Scoring**: Maps cosine similarity to strict scoring bands (Similarity >= 0.75 gives full marks, >= 0.50 gives partial marks).

## Prerequisites (Important!)
You must configure your OpenAI API Key inside an environment variables file. No local Tesseract engine binary installations are required!
- Create a `.env` file in the root of the `ai-service` directory and add:
  ```env
  OPENAI_API_KEY=your-openai-api-key-here
  PORT=8000
  ```

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

