



# Adobe Hackathon 2025 — Round 1A

## Challenge: Understand Your Document

---

## Problem Statement

This challenge was part of **Round 1A** of the **Adobe India Hackathon 2025** — themed *"Connecting the Dots"*.  
The objective was to **analyze unstructured PDF documents** and extract a **clean, structured outline**, identifying:

- The **document title**
- **Headings**, categorized as:
  - H1 (primary heading)
  - H2 (sub-heading)
  - H3 (sub-sub-heading)
- The **page number** where each heading occurs

The output needed to be formatted as **JSON** and include the full heading hierarchy with pagination.  
Importantly, the entire solution had to run:

- Offline
- On CPU-only hardware
- Under tight runtime and model-size constraints

---

## Our Approach

We adopted a **modular pipeline** combining **rule-based heuristics** with **lightweight machine learning**:

### 1. PDF Parsing

We used [`PyMuPDF`](https://pymupdf.readthedocs.io/en/latest/) (`fitz`) to extract raw layout data from each page:

- Font size
- Font name
- Text content
- Bounding box
- Font flags (bold, italic, etc.)
- Page number

This gave us a layout-rich, structured representation of the document text.

### 2. No Dataset? No Problem

Since no public dataset existed for this task, we created one ourselves using `generate_training_data.py`:

- Scraped structured websites (e.g., WHO, UN, NatGeo)
- Extracted HTML heading tags (`<title>`, `<h1>` to `<h6>`)
- Rendered the same pages as PDFs using Playwright
- Parsed both HTML and PDF
- Matched text blocks between them to auto-label data
- Normalized heading levels (e.g., lowest tag = H1, etc.)

This yielded a high-quality training dataset without manual annotation.

### 3. Heading Classification

We experimented with various lightweight classifiers using `train_heading_model.py`:

- MLPClassifier (optimized using `RandomizedSearchCV`)
- XGBoost
- LightGBM

**Input Features:**

- Font size (z-scored)
- Font name (one-hot encoded)
- Bounding box geometry
- Font flags (bold/italic)

The best model (in terms of accuracy and runtime) was saved as `best_model.pkl`.

### 4. Inference on New PDFs

Our final script, `extract_outline.py`, performs end-to-end inference:

- Parses new PDFs into text blocks
- Extracts layout features
- Applies pre-fitted encoders and scalers
- Predicts heading level for each block
- Builds and returns a hierarchical outline as JSON

**Sample Output:**

```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History of AI", "page": 3 }
  ]
}
```

##  Libraries & Models Used

###  Python Libraries:
| Library         | Purpose                                           |
|-----------------|---------------------------------------------------|
| `PyMuPDF (fitz)`| PDF parsing and layout/text extraction            |
| `scikit-learn`  | MLP classifier, feature encoding, scaling         |
| `xgboost`       | Gradient boosting ML model                        |
| `lightgbm`      | Light-weight gradient boosting ML model           |
| `pandas`        | Data handling and transformation                  |
| `numpy`         | Numerical computations                            |
| `playwright`    | Rendering web pages to PDF for dataset creation   |
| `joblib`        | Model serialization and deserialization           |

###  Models Tried:
| Model           | Description                                      | Used in Final |
|------------------|--------------------------------------------------|---------------|
| `MLPClassifier`  | Multi-layer perceptron (neural net)              |  Yes         |
| `XGBoost`        | Gradient boosting trees (accurate + fast)        |  Yes         |
| `LightGBM`       | Fast, lightweight boosting tree model            |  Yes         |



##  How to Build and Run

###  Build the Docker Image

Make sure you're in the root directory of the project, then run:

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .

```

### Run the Docker Container
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --
network none mysolutionname:somerandomidentifier

```




##  Potential Improvements

If model size and runtime constraints were relaxed, the following enhancements could improve both performance and accuracy:

| Area                 | Improvement Idea                                                                 |
|----------------------|----------------------------------------------------------------------------------|
| **Visual Features**  | Extract and classify **tables**, **figures**, and **captions**                   |
| **Semantic Modeling**| Use **Sentence Transformers** (e.g., `all-MiniLM`) for better semantic grouping  |
| **OCR Support**      | Add OCR fallback using `Tesseract` or `EasyOCR` for scanned documents            |
| **Deep Models**      | Switch to transformer-based models like `DistilBERT` or `LayoutLM`               |
| **Page Structure**   | Include **TOC detection**, **header/footer removal**, and **column detection**   |
| **TOC Alignment**    | Use heuristics or ML to align predicted headings with the actual Table of Contents |









