# System Enhancements Report

**Project:** Intelligent Assessment System (AI Service)  
**Date:** April 2026  
**Goal:** Address performance bottlenecks, enable batch processing, and introduce intelligent fallbacks to ensure smooth, robust operation in production environments.

---

## 1. Batch Upload and Background Processing Integration

**Problem:**  
The previously implemented API only supported single file uploads (`/grade`). For a lecturer looking to upload 50–100 examination scripts simultaneously, the frontend would face extensive timeouts and concurrency crashes.

**Solution:**  
- **Endpoint Added (`/batch-grade`)**: Implemented a new asynchronous batch processing API that accepts a `List[UploadFile]`.
- **FastAPI BackgroundTasks**: Integrated native background job delegation. The endpoint immediately responds with a tracking `job_id`, ensuring a frictionless UX on the frontend.
- **Status Polling (`/batch-status/{job_id}`)**: Created a lightweight tracking endpoint to allow the frontend to ping real-time completion statuses and retrieve array-aggregated results.

## 2. Event-Loop Unblocking for AI Operations

**Problem:**  
FastAPI is an asynchronous framework built on a single-threaded event loop. Running heavy CPU-bound functions (like `pytesseract` OCR processing and local Sentence-BERT model inferences) natively inside an `async def` route caused complete application starvation—blocking parallel user requests.

**Solution:**  
- Refactored the core workflow inside the route handling using `fastapi.concurrency.run_in_threadpool()`.
- Thread delegation completely shifts synchronous ML computations (OCR and Embedding operations) off the main loop, permitting highly scalable parallelism.

## 3. Rubric Computation Caching (Massive Optimization)

**Problem:**  
When looping through batch scripts, the architecture evaluated the Sentence-BERT vector generation for the predefined marking rubric over and over again per student script.

**Solution:**  
- Refactored the grading pipeline to strictly pre-compute rubric `embeddings` **once** at the start of a batch job iteration.  
- By storing rubric arrays securely in the scope before script mapping begins, we bypassed thousands of redundant matrix allocations. Expect heavy speedups for large student counts.

## 4. Automatic Student Identifier Extraction

**Problem:**  
Although scripts were correctly graded, the outputted format lacked programmatic identification logic. Without a link to who actually owned the score, the API integration was purely a sandbox metric.

**Solution:**  
- Developed the `extract_student_id()` utility inside `app/utils/text_preprocessing.py`.
- Introduced pattern matching via RegEx explicitly tuned toward predicting structural identifiers often found in Nigerian University conventions (e.g. `21/04CS023` or `19/MAC/011`). The identity is now properly mapped back to the structured database query via our updated Pydantic response models.

## 5. Intelligent Segmentation Fallback (`GPT-4o-Mini`)

**Problem:**  
Student layouts rarely confirm to standardized strict patterns. When `Q1` or `1(a)` breaks down due to bad handwriting or unexpected abbreviations, the core pattern matcher flagged the entire student prompt as `"Uncategorized"`, destroying chunked point similarity.

**Solution:**  
- Added an intelligent fallback layer into `app/services/segmentation_service.py`.  
- **Workflow:** When the regex chunker fails to hit expected patterns on long sequence strings, the engine automatically proxies the unstructured text through `gpt-4o-mini`. 
- **JSON Mode**: Using explicit prompt parameters and the `response_format={ "type": "json_object" }` flag, the fallback cleanly parses abstract student structures into standard indexed chunks, securing extraction accuracy. 
