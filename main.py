import os
import tempfile
import asyncio
import aiofiles
from transformers import AutoTokenizer, AutoModelForCausalLM
from fpdf import FPDF
import gradio as gr

# Initialize GPT-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

async def generate_cover_letter(job_title, job_description, hiring_manager, company_name, company_address, required_skills, company_trait, resume_text):
    resume_data = {
        "name": "[Your Full Name]",
        "address": "[Your Address]",
        "city": "[Your City]",
        "email": "[Your Email]",
        "phone": "+1234567890",
        "skills": "[Your Skills, e.g., Python, Flask, SQL]",
        "experience": "[Your Relevant Experience]",
        "achievements": "[Your Key Achievements]",
        "interests": "[Your Professional Interests, e.g., contributing to open-source projects]"
    }
    
    prompt = f"""
    {company_address}  
    
    {hiring_manager},
    
    I am writing to express my interest in the {job_title} position at {company_name}. With a strong background in {resume_data["skills"]}, particularly my proficiency in {resume_data["skills"]}, I am enthusiastic about the opportunity to apply my expertise to the innovative work being done at your company.
    
    Based on my resume, which includes {resume_data["experience"]}, I am confident that my skills and experiences make me a suitable candidate for this role.
    
    The job description mentions that you are looking for someone with experience in {required_skills}. In my previous roles, I have successfully {resume_data["achievements"]}.
    
    I am particularly drawn to {company_name} because of its commitment to {company_trait}. This commitment resonates with my own professional ethos and my passion for {resume_data["interests"]}.
    
    I am excited about the opportunity to contribute to {company_name} and help achieve your goals.
    
    Please find my resume attached for your review. I look forward to discussing how my skills and experiences can benefit {company_name}.
    
    Sincerely,
    {resume_data["name"]}
    """
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    max_length = 500
    generated_output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=0.7)
    cover_letter = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    
    return cover_letter

async def save_pdf_cover_letter(cover_letter_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Cover Letter", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, cover_letter_text)
    
    temp_file_path = tempfile.mktemp(suffix='.pdf')
    pdf.output(temp_file_path)
    
    return temp_file_path

async def process_resume_and_generate_cover_letter(resume, job_title, job_description, hiring_manager, company_name, company_address, required_skills, company_trait):
    resume_text = resume.name  # Directly get the text content
    cover_letter_text = await generate_cover_letter(job_title, job_description, hiring_manager, company_name, company_address, required_skills, company_trait, resume_text)
    pdf_path = await save_pdf_cover_letter(cover_letter_text)
    
    return pdf_path

resume_input = gr.File(label="Upload your resume (text format)")
job_title_input = gr.Textbox(label="Job Title")
job_description_input = gr.Textbox(label="Job Description")
hiring_manager_input = gr.Textbox(label="Hiring Manager")
company_name_input = gr.Textbox(label="Company Name")
company_address_input = gr.Textbox(label="Company Address- Remember to include the relevant address")
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
