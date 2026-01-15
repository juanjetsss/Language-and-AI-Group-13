import re 
import html 
import unicodedata
from typing import Optional, Set


import pandas as pd 
from tqdm import tqdm 

# Language detection 
try: 
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


import spacy

    

# 1. Regex
  
RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_HASHTAG = re.compile(r"#\w+")
RE_USERNAME = re.compile(r"(\bu/\w+\b|\b@\w+\b)", re.IGNORECASE)
RE_SUBREDDIT = re.compile(r"\br/\w+\b", re.IGNORECASE)

RE_DIGITS = re.compile(r"\d+")
RE_MULTISPACE = re.compile(r"\s+")

RE_ANSI = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")  
RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

WEIRD_CHARS = {"\ufeff", "\u200b", "�"}

GENDER_TERMS = r"(?:man|woman|male|female|guy|girl|dude|gal|lady|boy)"
GENDER_MOD = r"(?:(?:trans|cis)\s+)?"  

RE_SELF_ID_I_AM = re.compile(
    rf"\b(?:i\s*am|i['’]\s*m|i'm|im)\s+(?:a|an)?\s*{GENDER_MOD}{GENDER_TERMS}\b",
    re.IGNORECASE,
)
RE_SELF_ID_IDENTIFY = re.compile(
    rf"\b(?:i\s+(?:identify|identified)\s+as|i\s+identify\s+myself\s+as)\s+(?:a|an)?\s*{GENDER_MOD}{GENDER_TERMS}\b",
    re.IGNORECASE,
)
RE_SELF_ID_AS_A = re.compile(
    rf"\bas\s+(?:a|an)\s+{GENDER_MOD}{GENDER_TERMS}\b",
    re.IGNORECASE,
)
RE_SELF_ID_HERE = re.compile(
    rf"\b{GENDER_MOD}{GENDER_TERMS}\s+here\b",
    re.IGNORECASE,
)

# punctuation/emoticons to keep as tokens in the second dataset
RE_EMOT_SMILE = re.compile(r"(:-\)|:\))")
RE_EMOT_SAD = re.compile(r"(:-\(|:\()")

# Punctuation list: . , ! ? - () " ' : ; [ ]
RE_PUNCT_SIMPLE = re.compile(r"[.,!?():;\[\]\"]")

# Male vs female symbols
PUNCT_MAP = {
    ".": "punctdot",
    ",": "punctcomma",
    "!": "punctexclam",
    "?": "punctqmark",
    "(": "punctparenO",
    ")": "punctparenC",
    '"': "punctquote",
    ":": "punctcolon",
    ";": "punctsemicolon",
    "[": "punctbracketO",
    "]": "punctbracketC",
}

# 2. Text cleaning functions 

def remove_accents(text: str) -> str:
    #Replace accented characters
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))

def text_clean(text: str) -> str:
    if not isinstance(text, str): 
        return ""
    
    text = html.unescape(text)
    
    for ch in WEIRD_CHARS:
        text = text.replace(ch, " ")
        
    # Remove ANSI and control chars
    text = RE_ANSI.sub(" ", text)
    text = RE_CTRL.sub(" ", text)
    
    # Remove URLs, tags, mentions, hashtags, subreddits
    text = RE_URL.sub(" ", text)
    text = RE_HTML_TAG.sub(" ", text)
    text = RE_HASHTAG.sub(" ", text)
    text = RE_USERNAME.sub(" ", text)
    text = RE_SUBREDDIT.sub(" ", text) 
      
    # Remove digits
    text = RE_DIGITS.sub(" ", text)
    
    # Normalize accents
    text = remove_accents(text)
    
    # Normalize lower case and whitespace
    text = text.lower()
    text = RE_MULTISPACE.sub(" ", text).strip()
    
    return text

def neutralize_gender(text: str, token: str = "gender") -> str:
    # Replace explicit self-identification phrases
    if not isinstance(text, str):
        return ""

    text = RE_SELF_ID_I_AM.sub(f" {token} ", text)
    text = RE_SELF_ID_IDENTIFY.sub(f" {token} ", text)
    text = RE_SELF_ID_AS_A.sub(f" {token} ", text)
    text = RE_SELF_ID_HERE.sub(f" {token} ", text)

    text = RE_MULTISPACE.sub(" ", text).strip()
    return text

def punctuation_to_tokens(text: str) -> str: # Convert the chosen punctuation list into tokens
    if not isinstance(text, str):
        return ""

    # Emoticons
    text = RE_EMOT_SMILE.sub(" emotsmile ", text)
    text = RE_EMOT_SAD.sub(" emotsad ", text)

    def repl(m: re.Match) -> str:
        return f" {PUNCT_MAP[m.group(0)]} "
        
    text = RE_PUNCT_SIMPLE.sub(repl, text)

    # Apostrophes: avoids breaking don't / i'm
    text = re.sub(r"(?<![a-z])['’](?![a-z])", " punctapostrophe ", text)

    # Dashes: avoids breaking terms like "well-being"
    text = re.sub(r"-{2,}", " punctdash ", text)
    text = re.sub(r"(?<=\s)-(?=\s)", " punctdash ", text)

    text = RE_MULTISPACE.sub(" ", text).strip()
    return text

# 3. Language detection 

def detect_language(text: str) -> Optional[str]:
    # Returns language code or None if unavialable
    if not LANGDETECT_AVAILABLE:
        return None
    text = text.strip()
    if len(text) < 20: 
        return None
    try:
        return detect(text)
    except Exception:
        return None


# 4. Token filtering and lemmatization 

def spacy_tokenize_lemmatize(
    texts : pd.Series,
    spacy_model: str = "en_core_web_sm",
    min_token_len: int = 2,
    batch_size: int = 512,
    n_process: int = 1,
) -> pd.Series:
    try:
        nlp = spacy.load(spacy_model, disable=["parser", "ner"])
    except OSError as e:
        raise OSError(
            f"spaCy model '{spacy_model}' not found. Install it with:\n"
            f"  python -m spacy download {spacy_model}\n"
        ) from e
    
    cleaned_texts = []
    for doc in tqdm(
        nlp.pipe(texts.tolist(), batch_size=batch_size, n_process=n_process),
        total=len(texts),
        desc="spaCy processing",
    ):
        tokens = []
        for tok in doc:
            # Keep only words
            if not tok.is_alpha:
                continue
            # Remove stopwords
            if tok.is_stop:
                continue
            lemma = tok.lemma_.lower().strip()
            # Remove very short words 
            if len(lemma) < min_token_len: 
                continue
            tokens.append(lemma)
        
        cleaned_texts.append(" ".join(tokens))
        
    return pd.Series(cleaned_texts, index=texts.index)

def load_clean_and_filter(
    csv_path: str,
    allowed_languages: Optional[Set[str]] = {"en"},
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Handling missing or invalid labels
    df["female"] = pd.to_numeric(df["female"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["female"])
    df["female"] = df["female"].astype(int)
    df = df[df["female"].isin([0, 1])]
    print(f"[Labels] Dropped {before - len(df)} rows due to missing/invalid female label.")

    # Text cleaning
    df["post_raw"] = df["post"].astype(str)
    df["post_basic"] = df["post_raw"].apply(text_clean)
    df["post_basic"] = df["post_basic"].apply(neutralize_gender)

    # Language detection and filtering 
    if allowed_languages is not None:
        if not LANGDETECT_AVAILABLE:
            print("[Language] langdetect not installed -> skipping language filtering.")
            df["language"] = None
        else:
            tqdm.pandas(desc="Language detection")
            df["language"] = df["post_basic"].progress_apply(detect_language)

            lang_counts = df["language"].value_counts(dropna=False).head(10)
            print("[Language] Top detected languages (incl. None for 'undetected'):\n", lang_counts)

            before_lang = len(df)
            df = df[df["language"].isin(allowed_languages)]
            print(f"[Language] Dropped {before_lang - len(df)} rows not in {allowed_languages}.")

    return df.reset_index(drop=True)
    
# 5. Main processing function

def preprocess_dataset(
    csv_path: str,
    allowed_languages: Optional[Set[str]] = {"en"},
    spacy_model: str = "en_core_web_sm",
    min_token_len: int = 2,
    n_process: int = 1,
    keep_punct_tokens: bool = False,
    df_preloaded: Optional[pd.DataFrame] = None,   
) -> pd.DataFrame:
    
    if df_preloaded is None:
        df = load_clean_and_filter(csv_path, allowed_languages=allowed_languages)
    else:
        df = df_preloaded.copy()

    if keep_punct_tokens:
        df["post_basic"] = df["post_basic"].apply(punctuation_to_tokens)
                
 
    # spaCy cleaning
    df["post_clean"] = spacy_tokenize_lemmatize(
        df["post_basic"],
        spacy_model=spacy_model,
        min_token_len=min_token_len,
        n_process=n_process,
    )
    
    # Drop empty rows after cleaning
    before_empty = len(df)
    df = df[df["post_clean"].str.strip().astype(bool)]
    print(f"[Empty] Dropped {before_empty - len(df)} rows that became empty after cleaning.")
    
    keep_cols = ["auhtor_ID", "post_raw", "post_basic", "post_clean", "female"]
    if "language" in df.columns:
        keep_cols.append("language")

    df = df[keep_cols].reset_index(drop=True)
    print(f"[Done] Final dataset shape: {df.shape}")

    return df


# 6) Run code

if __name__ == "__main__":
    DATA_PATH = "gender.csv"

    # Load + basic clean + language filter 
    base_df = load_clean_and_filter(DATA_PATH, allowed_languages={"en"})

    # 1) First dataset (complete cleaning)
    df_clean = preprocess_dataset(
        csv_path=DATA_PATH,
        allowed_languages={"en"},
        spacy_model="en_core_web_sm",
        min_token_len=2,
        n_process=1,
        keep_punct_tokens=False,
        df_preloaded=base_df,   
    )
    df_clean[["auhtor_ID", "post_clean", "female"]].to_csv("gender_cleaned.csv", index=False)

    # 2) Second dataset (special symbols allowed for gender classification)
    df_clean_punct = preprocess_dataset(
        csv_path=DATA_PATH,
        allowed_languages={"en"},
        spacy_model="en_core_web_sm",
        min_token_len=2,
        n_process=1,
        keep_punct_tokens=True,
        df_preloaded=base_df,   
    )
    df_clean_punct[["auhtor_ID", "post_clean", "female"]].to_csv("gender_cleaned_punct.csv", index=False)

    print("\nSaved: gender_cleaned.csv and gender_cleaned_punct.csv")