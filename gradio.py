import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def generate_response(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True , max_length=512)

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature = 0.7,
            do_sample = True,
            pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response


def concept_explanattion(concept):
    prompt = f"Explain the concept of {concept} in detail with example"
    return generate_response(prompt, max_length=800)

def quiz_generator(concept):
   prompt = f"Generate 5 quiz on the concept of {concept} with different quies types(multiple choice, true/flase, short answer):, Give me the answer at the end:"
   return generate_response(prompt, max_length=1200)

#create gradio
with gr.Blocks() as app:
  gr.Markdown("# Educational AI assistant")
  with gr.Tabs():
    with gr.TabItem("Concept Explanattion"):
      concept = gr.Textbox(label="Enter a Concept", placeholder="e.g., Machine Learning")
      explain_btn = gr.Button("Explain")
      explain_output = gr.Textbox(label="Explanation",lines=10)
      explain_btn.click(concept_explanattion, inputs=concept, outputs=explain_output)


    with gr.TabItem("Quiz Generator"):
      concept = gr.Textbox(label="Enter a Topic", placeholder="e.g., Machine Learning")
      explain_btn = gr.Button("Generate Quiz")
      explain_output = gr.Textbox(label="Quiz_Qeustion And Answer",lines=15)
      explain_btn.click(quiz_generator, inputs=concept, outputs=explain_output)


app.launch()