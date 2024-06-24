from flask import Flask, request, render_template, send_file, redirect, url_for, flash
import os
import asyncio
import aiohttp
from transformers import AutoTokenizer, AutoModelForCausalLM
import tempfile
import logging
from fpdf import FPDF
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'supersecretkey'

# Initialize GPT-2 model and tokenizer (loaded asynchronously later)
tokenizer = None
model = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Asynchronous model loading function
async def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    logger.info("Model loaded successfully")

# Run model loading in a separate thread
threading.Thread(target=lambda: asyncio.run(load_model())).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        flash("No resume file part")
        return redirect(url_for('index'))
    file = request.files['resume']
    if file.filename == '':
        flash("No selected file")
        return redirect(url_for('index'))
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        flash(f"Resume uploaded successfully: {file.filename}.")
        return redirect(url_for('job_qualifications'))
    flash("Resume upload failed")
    return redirect(url_for('index'))

@app.route('/job_qualifications', methods=['GET', 'POST'])
async def job_qualifications():
    if request.method == 'POST':
        job_title = request.form.get('job_title', '')
        job_description = request.form.get('job_description', '')
        hiring_manager = request.form.get('hiring_manager', '')
        company_name = request.form.get('company_name', '')
        company_address = request.form.get('company_address', '')
        required_skills = request.form.get('required_skills', '')
        company_trait = request.form.get('company_trait', '')

        if not all([job_title, job_description, hiring_manager, company_name, company_address, required_skills, company_trait]):
            flash("Please fill in all fields.")
            return redirect(url_for('job_qualifications'))

        job = {
            "company_name": company_name,
            "job_title": job_title,
            "job_description": job_description,
            "hiring_manager": hiring_manager,
            "company_address": company_address,
            "required_skills": required_skills,
            "company_trait": company_trait
        }

        # Mock resume data (replace with actual extraction logic if needed)
        resume_data = {
            "name": "[input you name]",
            "address": "[input your address]",
            "city": "[input your city]",
            "email": "[your email]",
            "phone": "+1234567890",
            "skills": "[Change this with your actual relevant skills] Python, Flask, SQL",
            "experience": "[input your experience]",
            "achievements": "I managed to [input your projects]",
            "interests": "contributing to open-source projects"
        }

        cover_letter_text = await generate_cover_letter(job, resume_data)

        # Save cover letter to a temporary PDF file
        temp_file_path = save_pdf_cover_letter(cover_letter_text)

        flash("Your cover letter has been generated successfully.")
        return redirect(url_for('success', temp_file_path=temp_file_path))
    else:
        return render_template('job_qualifications.html')

@app.route('/success')
def success():
    temp_file_path = request.args.get('temp_file_path')
    return render_template('success.html', temp_file_path=temp_file_path)

@app.route('/download/<path:filename>', methods=['GET'])
def download(filename):
    return send_file(filename, as_attachment=True)

async def generate_cover_letter(job, resume_data):
    prompt = f"""
    Dear {job["hiring_manager"]},
    
    I am writing to express my interest in the {job["job_title"]} position at {job["company_name"]}. With a strong background in {resume_data["skills"]}, particularly my proficiency in {resume_data["skills"]}, I am enthusiastic about the opportunity to apply my expertise to the innovative work being done at your company.
    
    Based on my resume, which includes {resume_data["experience"]}, I am confident that my skills and experiences make me a suitable candidate for this role.
    
    The job description mentions that you are looking for someone with experience in {job["required_skills"]}. In my previous roles, I have successfully {resume_data["achievements"]}.
    
    I am particularly drawn to {job["company_name"]} because of its commitment to {job["company_trait"]}. This commitment resonates with my own professional ethos and my passion for {resume_data["interests"]}.
    
    I am excited about the opportunity to contribute to {job["company_name"]} and help achieve your goals.
    
    Sincerely,
    {resume_data["name"]}
    """
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text using the model
    max_length = 500  # Adjust the length as needed
    generated_output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=0.7)
    
    # Decode the generated text
    cover_letter = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    
    return cover_letter

def save_pdf_cover_letter(cover_letter_text):
    # Create a PDF instance
    pdf = FPDF()
    pdf.add_page()
    
    # Add a title
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Cover Letter", ln=True, align='C')
    pdf.ln(10)
    
    # Add the cover letter text
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, cover_letter_text)
    
    # Save the PDF to a temporary file
    temp_file_path = tempfile.mktemp(suffix='.pdf')
    pdf.output(temp_file_path)
    
    return temp_file_path

if __name__ == "__main__":
    app.run(debug=True)
