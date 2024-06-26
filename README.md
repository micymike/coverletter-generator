# coverletter-generator
for a flask UI run app.py
for gradio UI run main.py
project documentation- https://docs.google.com/document/d/1Pce_oTHquUgmHtPc6_Lm7MZmKY9jjVZkR5LEN_3gHB0/edit
## README

# Flask Cover Letter Generator

This Flask application allows users to upload their resume, fill in job qualifications, and generate a personalized cover letter. The cover letter is generated using a pre-trained GPT-2 model from Hugging Face's transformers library and can be downloaded as a PDF.

## Features
1. Upload a resume.
2. Fill in job qualifications.
3. Generate a personalized cover letter.
4. Download the generated cover letter as a PDF.

## Installation

### Prerequisites
- Python 3.6+
- pip

### Clone the repository
```bash
git clone https://github.com/yourusername/flask-cover-letter-generator.git
cd flask-cover-letter-generator
```

### Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download the GPT-2 model and tokenizer
The project uses the `gpt2` model from Hugging Face's transformers library. The required model and tokenizer will be downloaded automatically when you first run the application.

### Run the application
```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000/`.

## Project Structure
```
.
├── app.py                  # Main application file
├── requirements.txt        # Required Python packages
├── templates               # HTML templates
│   ├── index.html          # Home page template
│   ├── job_qualifications.html  # Job qualifications form template
│   └── success.html        # Success page template
├── static
│   └── styles.css          # CSS for styling the application
├── uploads                 # Directory for storing uploaded resumes
└── README.md               # This README file
```

## How to Use
1. **Home Page**: Upload your resume by selecting the file and clicking the "Upload" button.
2. **Job Qualifications**: After successfully uploading the resume, you will be redirected to a page to fill in the job qualifications.
3. **Generate Cover Letter**: Upon filling out the form and submitting it, the application will generate a personalized cover letter.
4. **Download PDF**: On the success page, you will have the option to download the generated cover letter as a PDF.

## HTML Templates
### index.html
The home page where users can upload their resume.

### job_qualifications.html
A form where users fill in job qualifications required to generate the cover letter.

### success.html
Displays a success message and provides a link to download the generated cover letter PDF.

## CSS Styling
The CSS file `styles.css` is used to style the HTML templates, centering the content and making the form elements look presentable.

## Dependencies
- Flask
- transformers
- fpdf

### requirements.txt
```
Flask==2.0.1
transformers==4.9.2
fpdf==1.7.2
torch==1.9.0
```

## Logging
Logging is configured to output informational messages, which can be helpful for debugging purposes.

## Acknowledgments
This project uses the GPT-2 model from Hugging Face's transformers library for generating cover letters. The project also uses Flask, a lightweight WSGI web application framework in Python, and FPDF for generating PDF files.

## License
This project is licensed under the MIT License.