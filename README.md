# Car Review RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **Retrieval-Augmented Generation (RAG)** system for summarizing car reviews.  
It allows users to ask free-text questions about car models and get concise summaries based on real user reviews, including strengths, common issues, and example quotes.  

---

## Features

- Summarizes car reviews from a large dataset.
- Handles free-text questions like:
  - "What do you think about the 2009 Nissan Xterra?"
  - "Compare the 2007 Toyota 4Runner with the 2009 Nissan Xterra."
- Supports comparisons between multiple car models (internally queries each model).
- Strictly based on real reviews; does not hallucinate specs or problems.
- Returns example user quotes for clarity.
- Built with **Python**, **Pinecone**, **SentenceTransformers**, and **Groq**.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/xsvmyx/car-review-rag.git
cd car-review-rag
```

2. Create a virtual environment (fish shell example):
```bash
python -m venv .venv
source .venv/bin/activate.fish
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file with your API keys:
```
PINECONE_API_KEY=your_pinecone_api_key
INDEX_v2=your_pinecone_index_name
GROQ_API_KEY1=your_groq_api_key
GROQ_API_KEY2=your_groq_api_key2
GROQ_API_KEY3=your_groq_api_key3
```

---

## Contributing Data

This project grows thanks to user contributions! If you have car reviews you’d like to share to improve the knowledge base, feel free to contribute.  

**Requirements:**  
- Each review should include at least the **car model** and **year**.  
- Including the **engine type** is a plus.  
- The text of the review itself is required.  

**How to contribute:**  
1. Submit your data via a **pull request** or open an **issue**.  
2. Once approved, your contribution will be added to the RAG system to make the bot smarter and more accurate.  

Every contribution helps the bot better summarize car reviews for everyone!