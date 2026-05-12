# NLP Practicals — Complete Beginner-Friendly Explanation

> [!TIP]
> Read this before your exam. Every line of every practical is explained below as if you're seeing Python and NLP for the first time.

---

## Table of Contents
1. [Practical 01 — Tokenization, Stemming & Lemmatization](#practical-01)
2. [Practical 02 — Bag of Words, TF-IDF & Word2Vec](#practical-02)
3. [Practical 03 — Text Preprocessing Pipeline](#practical-03)
4. [Practical 04 — Transformer Architecture](#practical-04)
5. [Practical 05 — Morphological Analysis](#practical-05)

---

## Practical 01
### Tokenization, Stemming, and Lemmatization

**What is this practical about?**
Computers don't understand sentences like humans do. Before a computer can work with text, it needs to break it into smaller pieces (tokens). This practical shows **5 different ways to break text** and **3 ways to reduce words to their base form**.

---

### The Imports (Lines 8–11)

```python
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer, MWETokenizer, TreebankWordTokenizer
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
import string
```

| Line | What it does |
|------|-------------|
| `import nltk` | Loads the **Natural Language Toolkit** — a Python library with tools for working with human language |
| `from nltk.tokenize import ...` | Loads 4 different tokenizer tools (they split text into pieces) |
| `from nltk.stem import ...` | Loads 3 tools that reduce words to their root/base form |
| `import string` | Loads a helper that knows all punctuation characters (`! , . ? ;` etc.) |

### Downloading Resources (Lines 14–17)

```python
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
```

NLTK needs some data files to work. Think of it like downloading a dictionary before you can look up words:
- **punkt** — rules for splitting sentences
- **wordnet** — a dictionary database (used for lemmatization)
- **omw-1.4** — multilingual word mappings

### The Input Text (Line 19)

```python
sentence = "Machine learning models are transforming healthcare! Let's explore NLP techniques :) #AI"
```

This is the sentence we'll break apart using different methods.

---

### 1. Whitespace Tokenizer (Line 22–23)

```python
ws_tokens = sentence.split()
print("1. Whitespace Tokens:", ws_tokens)
```

**How it works:** Simply splits wherever there's a **space**.

```
Input:  "Machine learning models are transforming healthcare!"
Output: ['Machine', 'learning', 'models', 'are', 'transforming', 'healthcare!', ...]
```

> [!NOTE]
> Notice `healthcare!` still has the `!` attached — this tokenizer doesn't care about punctuation, only spaces.

### 2. Punctuation-based Tokenizer (Line 26–27)

```python
clean_tokens = [w.strip(string.punctuation) for w in ws_tokens if w.strip(string.punctuation)]
```

**How it works:** Takes the whitespace tokens and **strips punctuation** from the edges of each word.

```
'healthcare!'  →  'healthcare'    (! removed)
':)'           →  ''              (empty, so it's dropped)
'#AI'          →  'AI'            (# removed)
```

### 3. Treebank Tokenizer (Lines 30–32)

```python
tb_tokenizer = TreebankWordTokenizer()
tb_tokens = tb_tokenizer.tokenize(sentence)
```

**How it works:** Uses rules from the **Penn Treebank** (a famous language dataset). It separates punctuation into its own token and splits contractions.

```
"Let's"  →  ['Let', "'s"]     (contraction split)
"!"      →  separate token
```

### 4. Tweet Tokenizer (Lines 35–37)

```python
tw_tokenizer = TweetTokenizer()
tw_tokens = tw_tokenizer.tokenize(sentence)
```

**How it works:** Designed for **social media text**. It keeps emoticons (`:)`) and hashtags (`#AI`) intact because they carry meaning on social media.

```
':)'   →  stays as ':)'    (emoticon preserved!)
'#AI'  →  stays as '#AI'   (hashtag preserved!)
```

### 5. Multi-Word Expression (MWE) Tokenizer (Lines 40–43)

```python
mwe = MWETokenizer([('machine', 'learning'), ('NLP', 'techniques')])
base_tokens = word_tokenize(sentence.lower())
mwe_tokens = mwe.tokenize(base_tokens)
```

**How it works:** You tell it which words **should stay together** as one token. "Machine learning" is a single concept, not two separate words.

```
['machine', 'learning', 'models']  →  ['machine_learning', 'models']
```

---

### 6. Porter Stemmer (Lines 46–48)

```python
porter = PorterStemmer()
porter_results = [porter.stem(tok) for tok in clean_tokens]
```

**What is Stemming?** Chopping off the end of a word to get a rough "root." It's fast but can give non-real words.

```
'transforming'  →  'transform'
'learning'      →  'learn'
'models'        →  'model'
'healthcare'    →  'healthcar'   ← Not a real word! But that's OK for stemming.
```

### 7. Snowball Stemmer (Lines 51–53)

```python
snowball = SnowballStemmer("english")
snowball_results = [snowball.stem(tok) for tok in clean_tokens]
```

**How it's different from Porter:** Snowball is an **improved version** of Porter. It handles some edge cases better (e.g., handles "Let's" → "let" cleanly where Porter gives "let'").

### 8. WordNet Lemmatizer (Lines 56–58)

```python
wnl = WordNetLemmatizer()
lemmas = [wnl.lemmatize(tok) for tok in clean_tokens]
```

**What is Lemmatization?** Unlike stemming, it gives you the **actual dictionary word** (called a "lemma").

```
'models'  →  'model'     (real word!)
'are'     →  'are'       (keeps it, because without context it doesn't know it's a verb)
```

> [!IMPORTANT]
> **Stemming vs Lemmatization — Exam Question!**
> - **Stemming** = fast, chops blindly, may give non-words (`healthcar`)
> - **Lemmatization** = slower, uses dictionary, always gives real words (`healthcare`)

---

## Practical 02
### Bag of Words, TF-IDF, and Word2Vec

**What is this practical about?**
Computers work with **numbers**, not words. This practical shows **4 ways to convert text into numbers** so that machine learning models can process it.

---

### The Dataset (Lines 19–23)

```python
documents = [
    "Deep learning powers modern NLP systems",
    "NLP systems process human language effectively",
    "Transformers revolutionized deep learning research"
]
```

Three sentences. We'll convert each one into a list of numbers.

---

### 1. Bag of Words (Lines 29–34)

```python
count_vec = CountVectorizer()
bow = count_vec.fit_transform(documents)
feature_names = count_vec.get_feature_names_out()
bow_array = bow.toarray()
```

**How it works:**
1. Find all **unique words** across all documents → these become column names
2. For each document, **count** how many times each word appears

**Example:**

| | deep | effectively | human | language | learning | modern | nlp | powers | process | research | revolutionized | systems | transformers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Doc 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| Doc 2 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| Doc 3 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 |

> [!NOTE]
> The word order is **completely lost** — hence "Bag" of Words (like throwing words into a bag).

### 2. Normalized Term Frequency (Lines 40–42)

```python
row_sums = bow_array.sum(axis=1, keepdims=True)
tf_normalized = bow_array / row_sums
```

**Problem with raw counts:** If Document A has 100 words and Document B has 10 words, raw counts unfairly favor A.

**Solution:** Divide each count by the **total words in that document** to get a proportion (0 to 1).

```
Doc 1 has 6 words. "deep" appears 1 time.
Normalized = 1/6 = 0.1667
```

### 3. TF-IDF (Lines 48–51)

```python
tfidf_vec = TfidfVectorizer()
tfidf_result = tfidf_vec.fit_transform(documents)
```

**TF-IDF = Term Frequency × Inverse Document Frequency**

The idea: A word that appears in **every** document (like "the", "is") is not useful. A word that appears in **only one** document is very informative.

- **TF** (Term Frequency) = how often the word appears in THIS document
- **IDF** (Inverse Document Frequency) = how RARE the word is across ALL documents

```
"NLP" appears in 2 out of 3 docs     → low IDF (common word)
"revolutionized" appears in 1 out of 3 → high IDF (rare, important word!)
```

> [!IMPORTANT]
> **Exam tip:** TF-IDF gives **higher scores to rare, meaningful words** and **lower scores to common words**.

### 4. Word2Vec (Lines 58–71)

```python
w2v_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=64,      # Each word becomes a list of 64 numbers
    window=4,            # Look at 4 neighboring words for context
    min_count=1,         # Include all words even if they appear once
    sg=1,                # Skip-gram method
    epochs=100           # Train 100 times over the data
)
```

**How it works:** Unlike BoW/TF-IDF which just count words, Word2Vec **learns the meaning** of words by looking at which words appear near each other.

```
"king" - "man" + "woman" ≈ "queen"   ← Word2Vec can do this!
```

Each word becomes a **vector** (list of 64 numbers). Words with similar meanings have similar vectors.

```python
w2v_model.wv['learning']         # Get the 64-number vector for "learning"
w2v_model.wv.most_similar('learning')  # Find words with similar meaning
```

> [!NOTE]
> **Skip-gram (`sg=1`)** predicts surrounding words from a center word. The alternative is **CBOW (`sg=0`)** which predicts the center word from surrounding words.

---

## Practical 03
### Text Preprocessing Pipeline

**What is this practical about?**
Real-world text is messy — it has capital letters, punctuation, useless words ("the", "is", "a"), and different forms of the same word ("running", "runs", "ran"). This practical **cleans text step by step** to make it ready for machine learning.

---

### The Pipeline (step by step):

```
Raw Text → Lowercase → Remove Punctuation → Tokenize → Remove Stopwords → Lemmatize → Encode Labels → TF-IDF
```

### Step 1: Raw Dataset (Lines 26–34)

```python
records = {
    "review": [
        "The movie was absolutely fantastic!",
        "I found the plot quite boring and predictable.",
        "Great performances by the lead actors in this film."
    ],
    "sentiment": ["positive", "negative", "positive"]
}
df = pd.DataFrame(records)
```

A **DataFrame** is like an Excel table:

| review | sentiment |
|--------|-----------|
| The movie was absolutely fantastic! | positive |
| I found the plot quite boring and predictable. | negative |
| Great performances by the lead actors in this film. | positive |

### Step 2: Text Cleaning (Lines 38–43)

```python
def preprocess(text):
    text = text.lower()                                              # Step A
    text = text.translate(str.maketrans('', '', string.punctuation))  # Step B
    return word_tokenize(text)                                       # Step C

df['tokens'] = df['review'].apply(preprocess)
```

| Step | Before | After |
|------|--------|-------|
| A: Lowercase | `"The Movie was FANTASTIC!"` | `"the movie was fantastic!"` |
| B: Remove punctuation | `"the movie was fantastic!"` | `"the movie was fantastic"` |
| C: Tokenize | `"the movie was fantastic"` | `['the', 'movie', 'was', 'fantastic']` |

### Step 3: Remove Stopwords (Lines 46–49)

```python
eng_stopwords = set(stopwords.words('english'))
df['filtered'] = df['tokens'].apply(
    lambda toks: [t for t in toks if t not in eng_stopwords]
)
```

**Stopwords** = extremely common words that don't carry meaning: *the, is, a, was, in, by, this, I, and, quite*

```
['the', 'movie', 'was', 'absolutely', 'fantastic']
                    ↓ remove stopwords
['movie', 'absolutely', 'fantastic']
```

### Step 4: Lemmatization (Lines 52–56)

```python
lem = WordNetLemmatizer()
df['lemmatized'] = df['filtered'].apply(
    lambda toks: [lem.lemmatize(t) for t in toks]
)
df['processed_text'] = df['lemmatized'].apply(lambda toks: " ".join(toks))
```

Reduces words to dictionary form: `performances → performance`, `actors → actor`

Then joins them back: `['movie', 'absolutely', 'fantastic']` → `"movie absolutely fantastic"`

### Step 5: Label Encoding (Lines 59–60)

```python
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['sentiment'])
```

Machine learning models need **numbers**, not words like "positive"/"negative":

```
"negative" → 0
"positive" → 1
```

### Step 6: TF-IDF (Lines 63–68)

```python
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['processed_text'])
```

Converts the cleaned text into the TF-IDF number matrix (same concept as Practical 02).

---

## Practical 04
### Transformer Architecture from Scratch

**What is this practical about?**
The **Transformer** is the most important model in modern NLP (it powers ChatGPT, Google Translate, etc.). This practical builds one from scratch using PyTorch.

> [!IMPORTANT]
> This is the most complex practical. The examiner will likely ask about the **concepts** (what each part does) more than the exact code.

---

### The Big Picture

```
Input (word IDs) → Embedding → + Positional Encoding → [Transformer Block × N] → Output
```

A Transformer Block contains:
```
Input → Multi-Head Attention → Add & Normalize → Feed Forward → Add & Normalize → Output
```

---

### Part 1: Positional Encoding (Lines 13–26)

```python
class SinusoidalPositionEncoding(nn.Module):
```

**Why do we need this?**
Unlike humans, the Transformer processes all words **at once** (not left-to-right). So it doesn't naturally know that "I" is the 1st word and "NLP" is the 5th word. Positional Encoding **adds position information** to each word.

**How?** It creates a unique pattern of sine and cosine waves for each position:
- Position 0 gets one pattern
- Position 1 gets a different pattern
- Position 2 gets another, and so on...

```python
pos_matrix[:, 0::2] = torch.sin(positions * denominator)   # Even columns = sine
pos_matrix[:, 1::2] = torch.cos(positions * denominator)   # Odd columns = cosine
```

These patterns are **added** to the word embeddings so the model knows word order.

### Part 2: Multi-Head Self-Attention (Lines 29–57)

```python
class MultiHeadSelfAttention(nn.Module):
```

**What is Attention?**
When you read "The cat sat on the **mat**", to understand "mat", you pay more **attention** to "sat" and "on" than to "The". Self-attention does the same thing mathematically.

**The 3 vectors — Query, Key, Value:**
Think of it like a search engine:
- **Query (Q)** = "What am I looking for?" (the current word asking a question)
- **Key (K)** = "What do I contain?" (every word advertising its content)
- **Value (V)** = "Here's my actual information" (the real data to retrieve)

```python
attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale   # How relevant is each word?
attn_probs = torch.softmax(attn_weights, dim=-1)               # Convert to probabilities (0-1)
context = torch.matmul(attn_probs, V)                          # Weighted sum of values
```

**Why "Multi-Head"?**
Instead of doing attention once, we do it **8 times** (8 heads) in parallel. Each head can focus on a different type of relationship:
- Head 1 might focus on grammar
- Head 2 might focus on meaning
- Head 3 might focus on nearby words

```python
self.head_dim = embed_dim // heads    # 128 ÷ 8 = 16 dimensions per head
```

### Part 3: Feed Forward Network (Lines 60–71)

```python
class PositionwiseFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),   # Expand: 128 → 512
            nn.GELU(),                          # Activation function
            nn.Dropout(dropout),                # Randomly turn off 10% neurons (prevents overfitting)
            nn.Linear(hidden_dim, embed_dim)    # Compress back: 512 → 128
        )
```

**What does this do?** After attention figures out which words are related, the FFN **processes** that information. It's like attention says "these words are connected" and FFN says "here's what that connection means."

- **GELU** = an activation function (like a switch that decides which signals to pass through)
- **Dropout** = randomly ignores some neurons during training to prevent memorization

### Part 4: Transformer Block (Lines 74–89)

```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        attn_out = self.self_attn(x, x, x)           # Step 1: Self-attention
        x = self.layernorm1(x + self.drop1(attn_out)) # Step 2: Add & Normalize
        ff_out = self.ffn(x)                          # Step 3: Feed Forward
        x = self.layernorm2(x + self.drop2(ff_out))   # Step 4: Add & Normalize
        return x
```

**Key concept — Residual Connection (`x + attn_out`):**
Instead of replacing `x` with the attention output, we **add** it. This is like saying "keep the original information AND add the new insights." This prevents the model from forgetting what a word originally meant.

**LayerNorm:** Keeps the numbers in a stable range so training doesn't explode.

### Part 5: Complete Transformer (Lines 92–108)

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, heads, ff_hidden, num_blocks):
        self.token_embed = nn.Embedding(vocab_size, embed_dim)  # Word → Vector
        self.pos_encode = SinusoidalPositionEncoding(...)        # + Position info
        self.blocks = nn.ModuleList([...])                       # Stack N blocks
        self.classifier = nn.Linear(embed_dim, vocab_size)      # Final prediction
```

**The full flow:**
```
Word IDs [42, 7, 156, ...]
    ↓ Embedding (lookup table: each ID → 128 numbers)
[0.2, -0.5, 0.8, ...]  × 128
    ↓ + Positional Encoding
[0.3, -0.3, 1.1, ...]  × 128
    ↓ Transformer Block 1
    ↓ Transformer Block 2
    ↓ Transformer Block 3
    ↓ Linear classifier (128 → 500)
Output: probability for each word in vocabulary
```

### Part 6: Test Run (Lines 111–124)

```python
VOCAB = 500        # Dictionary has 500 words
EMBED = 128        # Each word represented by 128 numbers
HEADS = 8          # 8 attention heads
FF_DIM = 512       # Feed-forward hidden size
BLOCKS = 3         # 3 transformer blocks stacked

test_input = torch.randint(0, VOCAB, (4, 12))   # 4 sentences, each 12 words long
test_output = transformer(test_input)
# Input:  (4, 12)       → 4 sentences of 12 word IDs
# Output: (4, 12, 500)  → for each word position, probability over 500 vocabulary words
```

---

## Practical 05
### Morphological Analysis

**What is this practical about?**
**Morphology** = the study of how words are formed. In English, you can create new words by adding prefixes (`un-happy`) or suffixes (`happy-ness`). This practical demonstrates those rules, plus stemming and lemmatization.

---

### Key Concepts

| Term | Meaning | Example |
|------|---------|---------|
| **Morpheme** | Smallest meaningful unit | `un` + `happy` + `ness` = 3 morphemes |
| **Prefix** | Added to the FRONT | `un-` + happy = **unhappy** |
| **Suffix** | Added to the END | happy + `-ness` = **happiness** |
| **Inflectional** | Changes form but NOT meaning class | run → run**ning** (still a verb) |
| **Derivational** | Changes the meaning class | happy (adj) → happi**ness** (noun) |

---

### Part 1: Morphological Rules Function (Lines 18–48)

```python
def apply_morphology(word):
    transformations = []

    # Rule 1: Prefix 'un-' (negation)
    transformations.append(("Prefix 'un-'", word, "un" + word))
    # happy → unhappy

    # Rule 2: Prefix 're-' (again)
    transformations.append(("Prefix 're-'", word, "re" + word))
    # make → remake

    # Rule 3: Suffix '-ing' (with 'e' dropping rule)
    if word.endswith('e'):
        derived = word[:-1] + "ing"    # make → mak + ing = making
    else:
        derived = word + "ing"         # happy → happying

    # Rule 4: Suffix '-ness' (with 'y' → 'i' rule)
    if word.endswith('y'):
        derived = word[:-1] + "iness"  # happy → happ + iness = happiness
    else:
        derived = word + "ness"        # beautiful → beautifulness

    # Rule 5: Suffix '-er'
    # happy → happyer

    # Rule 6: Suffix '-ly'
    # beautiful → beautifully
```

> [!NOTE]
> **The 'e' dropping rule**: When adding `-ing` to a word ending in silent `e`, drop the `e` first:
> `make` → `mak` + `ing` = `making` (NOT `makeing`)
>
> **The 'y' to 'i' rule**: When adding `-ness` to a word ending in `y`, change `y` to `i`:
> `happy` → `happi` + `ness` = `happiness` (NOT `happyness`)

### Part 2: Stemming & Lemmatization (Lines 51–55)

```python
def analyze_word(word):
    stem = stemmer.stem(word)          # Chop the ending
    lemma = lemmatizer.lemmatize(word) # Look up dictionary form
    return stem, lemma
```

| Word | Stem (Porter) | Lemma (WordNet) |
|------|:---:|:---:|
| happy | happi | happy |
| running | run | running |
| make | make | make |
| beautiful | beauti | beautiful |

### Part 3: Testing Multiple Words (Lines 58–72)

```python
test_words = ["happy", "running", "make", "beautiful"]

for word in test_words:
    rules = apply_morphology(word)     # Apply all 6 rules
    stem, lemma = analyze_word(word)   # Get stem and lemma
```

Instead of asking for user input, we test 4 different words that each demonstrate different morphology rules.

---

## Quick Revision Cheat Sheet

| Practical | One-Line Summary |
|-----------|-----------------|
| **01** | 5 ways to split text + 3 ways to find root words |
| **02** | 4 ways to convert text → numbers (BoW, TF, TF-IDF, Word2Vec) |
| **03** | Full cleaning pipeline: lowercase → remove punctuation → tokenize → remove stopwords → lemmatize → encode labels → TF-IDF |
| **04** | Build a Transformer: Positional Encoding + Multi-Head Attention + FFN + Stack them |
| **05** | Apply prefix/suffix rules to words + stemming vs lemmatization |

> [!TIP]
> **Most likely exam questions:**
> 1. Difference between stemming and lemmatization
> 2. What is TF-IDF and why is it better than BoW?
> 3. What is self-attention in Transformers?
> 4. What are stopwords? Give examples.
> 5. What is Word2Vec? How is Skip-gram different from CBOW?
> 6. What is morphology? Explain inflectional vs derivational.
