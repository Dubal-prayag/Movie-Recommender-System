# Movie Recommendation System


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import nltk


nltk.download('punkt')


movies = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['Batman Begins', 'The Dark Knight', 'Interstellar', 'Inception', 'Tenet'],
    'tags': [
        'action hero gotham city',
        'joker batman chaos crime',
        'space time travel blackhole',
        'dream subconscious heist thriller',
        'time inversion spy thriller'
    ]
}

df = pd.DataFrame(movies)


ps = PorterStemmer()
def stem_text(text):
    return " ".join([ps.stem(word) for word in text.split()])
df['tags'] = df['tags'].apply(stem_text)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()


similarity = cosine_similarity(vectors)


def recommend(movie_name):
    movie_name = movie_name.lower()
    titles_lower = df['title'].str.lower()
    
    if movie_name not in titles_lower.values:
        print("\n🚨 Movie not found! Try: Inception, Tenet, Batman Begins, etc.")
        return
    
    idx = titles_lower[titles_lower == movie_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:4]  # Top 3 recommendations
    
    print(f"\nRecommended movies for '{df.iloc[idx]['title']}':")
    for i in scores:
        print(f"- {df.iloc[i[0]]['title']}")


recommend("Inception")
