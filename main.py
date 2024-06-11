from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load the model and tokenizer
model_output_path = "C:\\Users\\Pasindu\\chatNew\\model_output"
model = GPT2LMHeadModel.from_pretrained(model_output_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_output_path)

def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data['prompt']
    response = generate_response(prompt)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
