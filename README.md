# trimister-2-ml-Assignment

This repository contains ML assignments along with skill setups for PDF processing, document handling, and Jupyter-based data analysis.

## Skills Installed

| Skill | Packages | Description |
|-------|----------|-------------|
| **PDF Skill** | PyPDF2, pdfplumber, pymupdf | Read and extract text/metadata from PDF files |
| **Docs Skill** | python-docx, docx2txt | Create and read Microsoft Word documents |
| **Jupyter Data Analysis Skill** | pandas, numpy, matplotlib, seaborn, scikit-learn, scipy | Full data analysis and machine learning toolkit |

## Getting Started

### 1. Install all dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter Notebook

```bash
jupyter notebook skills_setup.ipynb
```

The notebook `skills_setup.ipynb` walks through each skill with working examples:
- **PDF Skill** – extracts text and metadata from `Graded Assignment ML 1.pdf`
- **Docs Skill** – creates and reads a sample Word document
- **Jupyter Data Analysis Skill** – loads the Iris dataset, visualises it, and trains a Logistic Regression classifier

## Files

| File | Description |
|------|-------------|
| `requirements.txt` | All required Python packages |
| `skills_setup.ipynb` | Jupyter notebook demonstrating all three skills |
| `Graded Assignment ML 1.pdf` | Graded ML assignment (PDF) |
