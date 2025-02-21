from flask import Flask, request, jsonify
from ctransformers import AutoModelForCausalLM
from flask_cors import CORS
from transformers import AutoModel

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Requests

# Load chatbot model
#model_path = "./models/Mistral-7B-Instruct-v0.1.Q2_K.gguf"
# Load model directly

model_path= AutoModel.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF")
llm = AutoModelForCausalLM.from_pretrained(model_path, model_type="mistral")  # Explicit model type

# Store chat history
chat_history = []

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"response": "Please enter a message."})

    # Prepare chat history for context
    context = "\n".join([f"{role}: {text}" for role, text in chat_history]) + f"\nUser: {user_input}\nAI:"

    # Generate chatbot response
    response = llm(context, max_new_tokens=150, temperature=0.7, top_p=0.9, repetition_penalty=1.2, stream=False).strip()

    # Append user & AI messages to history (limit to last 10 exchanges)
    chat_history.append(("User", user_input))
    chat_history.append(("AI", response))
    chat_history = chat_history[-10:]  # Keep only the last 10 exchanges

    return jsonify({"response": response})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
