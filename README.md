# Audio Notes Application

This application transcribes audio notes and stores them in a vector database for searching.

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2. Create a Conda environment:
    ```bash
    conda create --name audio-notes python=3.8
    conda activate audio-notes
    ```

3. Install dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4. Add your `.env` file with your `OPENAI_API_KEY`.

## Usage

Run the application with Streamlit:
```bash
streamlit run app.py
