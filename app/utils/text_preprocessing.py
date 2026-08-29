# src/ai-service/app/utils/text_preprocessing.py
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def clean_ocr_output(raw: str) -> str:
    """
    Strips GPT-4 Vision conversational preambles and markdown fences
    from OCR output before downstream processing.
    """
    # Remove leading conversational sentences up to and including first colon+newline
    raw = re.sub(r'^.*?(?:structure|follows|text)[^:]*:\s*\n+', '', raw, flags=re.IGNORECASE | re.DOTALL)

    # Strip markdown code fences  (``` or ```)
    raw = re.sub(r'```[a-z]*\n?', '', raw)

    # Strip leading/trailing whitespace
    return raw.strip()

def extract_student_id(text: str) -> str:
    """
    Attempts to extract a Nigerian University Matriculation Number or Student ID
    from the raw OCR text. Handles common OCR errors and spacing variations.
    Typical formats: IFS/24/9293, IFS/20/4986, IFS12419279, 1FS/24/9293, etc.
    """
    if not text:
        return "UNKNOWN"

    # Normalize: uppercase, collapse spaces around separators
    normalized = text.upper()
    normalized = re.sub(r'\s*/\s*', '/', normalized)  # "I FS / 24 / 92" -> "I FS/24/92"
    normalized = re.sub(r'\s+', ' ', normalized)      # collapse multiple spaces

    # Pattern 1: standard slash-separated format
    # e.g. IFS/24/9293, IFS/20/4986, CSC/21/1234
    pattern1 = re.compile(
        r'(?:MATRIC|ID|NO\.?|NUMBER)?\s*[:\-]?\s*'
        r'((?:[A-Z]{2,5}/)?\d{2,4}/[A-Z]{2,5}/\d{3,6})'
    )
    # Pattern 2: no-slash format (e.g. IFS12419279)
    pattern2 = re.compile(
        r'(?:MATRIC|ID|NO\.?|NUMBER)?\s*[:\-]?\s*'
        r'([A-Z]{2,5}\d{6,10})'
    )
    # Pattern 3: simple numeric with optional prefix (e.g. 21/04CS023)
    pattern3 = re.compile(
        r'(?:MATRIC|ID|NO\.?|NUMBER)?\s*[:\-]?\s*'
        r'(\d{2}/[A-Z0-9]+)'
    )

    for pattern in [pattern1, pattern2, pattern3]:
        match = pattern.search(normalized)
        if match:
            raw_id = match.group(1)
            # Fix common OCR errors
            raw_id = raw_id.replace(' ', '')           # remove any remaining spaces
            raw_id = re.sub(r'^1([A-Z])', r'I\1', raw_id)  # leading 1 -> I
            return raw_id

    return "UNKNOWN"
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data safely
try:
    nltk.download('punkt', quiet=False)
    nltk.download('punkt_tab', quiet=False)
    nltk.download('stopwords', quiet=False)
    nltk.download('wordnet', quiet=False)
except Exception as e:
    print(f"Warning: Failed to download NLTK data: {e}")

def preprocess_text(text: str) -> str:
    """
    Cleans textual answers by lowercasing, removing punctuation, 
    tokenization, stop-word removal, lemmatization, and standardizing whitespace
    to prepare for embedding.
    This step helps the embedding model focus on semantics rather than syntax.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove all punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # Fallback if punkt is missing
        tokens = text.split()
    
    # Stop-word removal
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    except LookupError:
        pass # Skip stop words if corpus missing

    # Lemmatization
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    except LookupError:
        pass # Skip lemmatization if wordnet missing
        
    return ' '.join(tokens)
