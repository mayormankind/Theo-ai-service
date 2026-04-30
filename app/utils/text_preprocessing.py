import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def extract_student_id(text: str) -> str:
    """
    Attempts to extract a Nigerian University Matriculation Number or Student ID
    from the raw OCR text. Typical formats: 21/04CS023, 19/MAC/011, etc.
    """
    # Regex for standard matric numbers: optional prefix, digits/letters separated by slashes
    pattern = re.compile(r'(?i)(?:matric|id|no\.?)?\s*[:\-]*\s*([0-9]{2,4}/[a-z]{2,5}/[0-9]{3,4}|[0-9]{2}/[a-z0-9]+)')
    match = pattern.search(text)
    if match:
        return match.group(1).upper()
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
