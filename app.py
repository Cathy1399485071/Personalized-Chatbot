import os
import openai
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from langchain.document_loaders import PyPDFLoader

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure your OpenAI API key
openai.api_key = 'openai-key'

# Store selected items, generated DAG, and topics
selected_items = []
dag_data = []

def extract_topics_from_document(pages):
    for page in pages:
        document_text = page.page_content
    

    # Prompt to instruct GPT to extract key topics in a hierarchical format
    prompt = f"Extract key topics from the following document. Structure them as main topics and subtopics as needed:\n\n{document_text}\n\nReturn a list of topics in hierarchical format."

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    # Process response to get a list of topics
    topics = response.choices[0].message['content'].strip().split('\n')
    return [{"id": topic} for topic in topics if topic]  # Format as list of topics

def write_to_vectordb (pages):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
    )
    splits = text_splitter.split_documents(pages)

    persist_directory = 'docs/chroma/' #it will create a docs/chroma each time running it
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    print(vectordb._collection.count())


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # load file into pages
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        
        # do vector embedding and store into database
        write_to_vectordb(pages)

        # extract text from the book
        global dag_data
        dag_data = extract_topics_from_document(pages)

        return jsonify({"status": "success", "dag": dag_data}), 200
    return jsonify({"status": "error"}), 400

@app.route('/select_item', methods=['POST'])
def select_item():
    item_name = request.json.get('item_name')
    if item_name:
        selected_items.append(item_name)
        return jsonify({"status": "success", "item_name": item_name}), 200
    return jsonify({"status": "error"}), 400

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    quiz_questions = [{"question": f"What is {topic['id']}?", "options": ["A", "B", "C", "D"], "answer": "A"} for topic in selected_items]
    return jsonify(quiz_questions), 200

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json.get('question')
    if question:
        # Mock response for LLM
        response = "Mock answer from LLM"
        return jsonify({"answer": response}), 200
    return jsonify({"status": "error"}), 400

if __name__ == '__main__':
    app.run(debug=True)

