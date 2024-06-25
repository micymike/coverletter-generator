import os
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config
from fpdf import FPDF
import pdfplumber
import gradio as gr
import torch

# Initialize GPT-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
config = GPT2Config.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_config(config)

MAX_INPUT_LENGTH = 900  # Maximum length for the input prompt to ensure it doesn't exceed the limit

def truncate_text(text, max_length):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.decode(tokens)

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def generate_cover_letter(job_title, job_description, hiring_manager, company_name, company_address, required_skills, company_trait, resume_text):
    # Truncate resume text if it exceeds the maximum length
    resume_text = truncate_text(resume_text, MAX_INPUT_LENGTH)
    
    prompt = f"""
    Dear {hiring_manager},

    I am writing to apply for the {job_title} position at {company_name}. I am highly enthusiastic about the opportunity to contribute to your team, particularly given your focus on {company_trait}.

    With a strong background in {required_skills}, I have developed excellent skills and experience in {resume_text}. I am confident in my ability to leverage my expertise to benefit {company_name}.

    In my previous roles, I have successfully {resume_text}. My professional ethos aligns with {company_name}'s commitment to {company_trait}, and I am passionate about bringing my skills and experience to your team.

    Thank you for considering my application. I look forward to the possibility of contributing to {company_name}.

    Sincerely,
    [Your Name]

    Additional Information:
    {job_description}
    """

    print("Prompt for GPT-2:\n", prompt)  # Debug print

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Check if pad_token_id is available
    if tokenizer.pad_token_id is not None:
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    else:
        # Create attention mask using a default value
        attention_mask = input_ids.new_ones(input_ids.shape, dtype=torch.long)

    print("Input IDs shape:", input_ids.shape)
    print("Attention Mask shape:", attention_mask.shape)
    print("Input IDs data type:", input_ids.dtype)
    print("Attention Mask data type:", attention_mask.dtype)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Use the global model variable
    global model
    model = model.to(device)

    max_new_tokens = 300  # Limit the number of new tokens generated
    generated_output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, num_return_sequences=1, temperature=0.7, do_sample=True)
    cover_letter = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    
    print("Generated Cover Letter:\n", cover_letter)  # Debug print

    return cover_letter

def save_pdf_cover_letter(cover_letter_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, cover_letter_text)
    
    temp_file_path = tempfile.mktemp(suffix='.pdf')
    pdf.output(temp_file_path)
    
    return temp_file_path

def process_resume_and_generate_cover_letter(resume, job_title, job_description, hiring_manager, company_name, company_address, required_skills, company_trait):
    resume_path = resume.name  # Get the file path of the uploaded resume
    resume_text = extract_text_from_pdf(resume_path)  # Extract text from the PDF
    cover_letter_text = generate_cover_letter(job_title, job_description, hiring_manager, company_name, company_address, required_skills, company_trait, resume_text)
    pdf_path = save_pdf_cover_letter(cover_letter_text)
    
    return pdf_path

resume_input = gr.File(label="Upload your resume (PDF format)")
job_title_input = gr.Textbox(label="Job Title")
job_description_input = gr.Textbox(label="Job Description")
hiring_manager_input = gr.Textbox(label="Hiring Manager")
company_name_input = gr.Textbox(label="Company Name")
company_address_input = gr.Textbox(label="Company Address")
required_skills_input = gr.Textbox(label="Required Skills")
company_trait_input = gr.Textbox(label="Company Trait")

output = gr.File(label="Download Cover Letter")

iface = gr.Interface(
    fn=process_resume_and_generate_cover_letter,
    inputs=[
        resume_input,
        job_title_input,
        job_description_input,
        hiring_manager_input,
        company_name_input,
        company_address_input,
        required_skills_input,
        company_trait_input
    ],
    outputs=output,
    title="Cover Letter Generator",
    description="Upload your resume and fill in the job qualifications to generate a personalized cover letter."
)

if __name__ == "__main__":
    iface.launch()
