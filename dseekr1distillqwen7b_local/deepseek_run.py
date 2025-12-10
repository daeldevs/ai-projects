from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re

model_path = r""

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model on GPU (4-bit)...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="cuda"
)
print("Model loaded!\n")

def extract_final(text):
    """Extract <final> if present, fallback to raw text"""
    if "<final>" in text:
        try:
            return text.split("<final>")[1].split("</final>")[0].strip()
        except:
            return text
    return text.strip()

def clean_tags(text):
    """Remove <|system|>, <|user|>, <|assistant|> tags"""
    return re.sub(r"<\|/?(system|user|assistant)\|>", "", text).strip()

# System prompt, added only once
system_prompt = (
    "You are a helpful AI assistant. Always reply concisely and clearly. "
    "Do not repeat yourself. Ignore <think> steps and only output the final answer."
)

print("Chat ready. Type 'exit' to quit.\n")

last_assistant = ""  # store only raw reply for minimal context

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    # Prompt: system prompt only first turn
    if last_assistant == "":
        prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_input}\n<|assistant|>\n"
    else:
        # Minimal context: last assistant + current user, no system
        prompt = f"<|assistant|>\n{last_assistant}\n<|user|>\n{user_input}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = extract_final(decoded)
    reply = clean_tags(reply)

    print("\nAI:", reply, "\n")

    # Store cleaned reply for next turn
    last_assistant = reply
