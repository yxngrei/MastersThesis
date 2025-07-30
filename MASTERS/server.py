from flask import Flask, request, jsonify
from gensim.models import Word2Vec

app = Flask(__name__)

model = Word2Vec.load("word2vec_fine_tuned.model")

@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.get_json()
    chord = data.get("chord", "C")
    try:
        suggestions = model.wv.most_similar(chord, topn=5)
        return jsonify([s[0] for s in suggestions])
    except KeyError:
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)