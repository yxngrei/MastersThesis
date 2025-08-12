from flask import Flask, request, jsonify
from flask_cors import CORS
import gensim
from gensim.models import Word2Vec

app = Flask(__name__)
CORS(app)

# Load your existing model
MODEL_PATH = r"MASTERS\word2vec_fine_tuned.model"  # Use raw string for Windows paths
model = Word2Vec.load(MODEL_PATH)

@app.route('/api/suggest/<chord_name>', methods=['GET'])
def suggest_chords(chord_name):
    """Get chord suggestions from your Gensim model based on a single chord name in the URL"""
    try:
        # Optional query parameter for number of suggestions
        num_suggestions = request.args.get('num_suggestions', default=5, type=int)

        if chord_name.lower() in ["", "null", "none"]:
            # Return common starting chords
            suggestions = [
                {'chord': 'C', 'score': 0.9},
                {'chord': 'G', 'score': 0.8},
                {'chord': 'Am', 'score': 0.7},
                {'chord': 'F', 'score': 0.6}
            ]
        else:
            try:
                similar = model.wv.most_similar(chord_name, topn=num_suggestions)
                suggestions = [{'chord': chord, 'score': score} for chord, score in similar]
            except KeyError:
                # Chord not in vocabulary, return fallback
                suggestions = [{'chord': 'C', 'score': 0.5}]

        return jsonify({
            'suggestions': suggestions,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/similarity', methods=['POST'])
def chord_similarity():
    """Calculate similarity between chords"""
    try:
        data = request.json
        chord1 = data.get('chord1')
        chord2 = data.get('chord2')
        
        similarity = model.wv.similarity(chord1, chord2)
        
        return jsonify({
            'similarity': float(similarity),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
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
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


if __name__ == '__main__':
    print(f"Loading model from {MODEL_PATH}")
    print(f"Model loaded with {len(model.wv.key_to_index)} chords")
    print("Starting Flask server...")
    app.run(debug=True, port=5000)
