from flask import Flask, request, jsonify
from flask_cors import CORS
from gensim.models import Word2Vec
import os

app = Flask(__name__)
CORS(app)

# Path to the model file
MODEL_PATH = os.path.join("MASTERS", "word2vec_fine_tuned.model")

# Load model once at startup
print(f"Loading model from {MODEL_PATH}...")
model = Word2Vec.load(MODEL_PATH)
print(f"Model loaded with {len(model.wv.key_to_index)} chords")

@app.route('/api/suggest/<chord_name>', methods=['GET'])
def suggest_chords(chord_name):
    """Get chord suggestions based on a chord name in the URL"""
    try:
        num_suggestions = request.args.get('num_suggestions', default=5, type=int)

        if not chord_name or chord_name.lower() in ["null", "none"]:
            suggestions = [
                {'chord': 'C', 'score': 0.9},
                {'chord': 'G', 'score': 0.8},
                {'chord': 'Am', 'score': 0.7},
                {'chord': 'F', 'score': 0.6}
            ]
        else:
            try:
                similar = model.wv.most_similar(chord_name, topn=num_suggestions)
                suggestions = [{'chord': chord, 'score': float(score)} for chord, score in similar]
            except KeyError:
                suggestions = [{'chord': 'C', 'score': 0.5}]

        return jsonify({'suggestions': suggestions, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/similarity', methods=['POST'])
def chord_similarity():
    try:
        data = request.json
        chord1 = data.get('chord1')
        chord2 = data.get('chord2')
        similarity = model.wv.similarity(chord1, chord2)
        return jsonify({'similarity': float(similarity), 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    try:
        vocab_size = len(model.wv.key_to_index)
        sample_words = list(model.wv.key_to_index.keys())[:10]
        return jsonify({
            'vocab_size': vocab_size,
            'sample_chords': sample_words,
            'vector_size': model.wv.vector_size,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
