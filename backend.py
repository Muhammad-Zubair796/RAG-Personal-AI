### backend.py ###
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.search import RAGSearch

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable cross-origin requests

# Initialize RAG system
print("[INFO] Initializing RAG system. This may take a few seconds...")
rag = RAGSearch()
print("[INFO] RAG system ready.")

# Section-wise static data for frontend cards
sections_data = {
    "Games": [
        "Create sprites and animate them using pure JavaScript without frameworks.",
        "Develop games using Unity.",
        "Built 4 interactive games."
    ],
    "ML/AI": [
        "ML/DL models including ANN, RNN, CNN, NLP.",
        "Developed solutions like flower classification, customer churn prediction, sentiment analysis, bank fraud detection.",
        "Experience with YOLO object detection, OpenCV, FastAPI, Flask, deploying models to Azure Cloud."
    ],
    "Web/Mobile": [
        "Full-stack web development including React, Next.js, Node.js, HTML, CSS.",
        "Created Android apps including Object Detection app, Tetromino Game, Weather App, ATS App.",
        "Databases: MySQL and data-driven applications."
    ],
    "Digital Marketing": [
        "SEO, SEM, Google Ads, Meta Ads, local SEO strategies.",
        "WordPress / WooCommerce development.",
        "Campaign optimization and ROI-focused strategies."
    ],
    "Graphics/Animation": [
        "2D animation, motion graphics, logo animation, video editing (After Effects, Photoshop, Illustrator, Moho).",
        "Design PDFs, PPTs, banners, marketing collaterals."
    ],
    "Projects": [
        "Multiple ML/DL solutions and enterprise automation systems.",
        "Full-stack web applications, Chrome extensions, React/Next.js projects.",
        "Android apps including Object Detection, Tetromino Game, Weather App, ATS App.",
        "4 Unity-based games and multiple JavaScript games."
    ]
}

# Serve frontend page
@app.route('/')
def home():
    return render_template('index.html')

# Query endpoint
@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get('query', '').strip()
    if not query_text:
        return jsonify({"answer": "Please provide a query."})

    try:
        answer = rag.search_and_summarize(query_text, top_k=5)
        if not isinstance(answer, str):
            answer = str(answer)  # Ensure JSON serializable
    except Exception as e:
        print(f"[ERROR] RAG query failed: {e}")
        answer = "Error: Could not process the query."

    return jsonify({"answer": answer})

# Sections endpoint
@app.route('/sections', methods=['GET'])
def get_sections():
    return jsonify(sections_data)

# Run server
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Starting Flask server on port {port} ...")
    app.run(host='0.0.0.0', port=port, debug=True)

