import tweepy
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
import seaborn as sns
import json

#token1: AAAAAAAAAAAAAAAAAAAAALnO0AEAAAAAb6XI24cj5QgVjOj6OmBwFWGfO98%3DJjFgaeSFuwEvI7odbWbHeTb1fA9SfQmzpMxLTPSigmUnbtUmH0
#token2: AAAAAAAAAAAAAAAAAAAAAAjP0AEAAAAAAweK0AyXBXC%2By3aULOcHMpkA3cU%3D2jkiLsyrlY7Cd0AGWxyGLKyt2XW3JnTaj9u67lj2W0kx8OWRfA
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAALnO0AEAAAAAb6XI24cj5QgVjOj6OmBwFWGfO98%3DJjFgaeSFuwEvI7odbWbHeTb1fA9SfQmzpMxLTPSigmUnbtUmH0"
cliente = tweepy.Client(bearer_token=BEARER_TOKEN)

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def aplicar_vader(texto):
    scores = sia.polarity_scores(texto)
    return scores['compound']

query = "#educacao lang:pt"
tweets = cliente.search_recent_tweets(query=query, max_results=10)
print(tweets)

dados = pd.DataFrame([tweet.text for tweet in tweets.data], columns=["texto"])

traduzir = GoogleTranslator(source='pt', target='en')
dados["texto_en"] = dados["texto"].apply(lambda x: traduzir.translate(x))

dados["vader_pt"] = dados["texto"].apply(aplicar_vader)
dados["vader_en"] = dados["texto_en"].apply(aplicar_vader)

sentiment_en = pipeline("sentiment-analysis")
dados["bert_en"] = dados["texto_en"].apply(lambda x: sentiment_en(x)[0]['score'] * (1 if sentiment_en(x)[0]['label'] == 'POSITIVE' else -1))

sentiment_pt = pipeline("sentiment-analysis", model="neuralmind/bert-base-portuguese-cased")
dados["bertimbau_pt"] = dados["texto"].apply(lambda x: sentiment_pt(x)[0]['score'] * (1 if sentiment_pt(x)[0]['label'] == 'POSITIVE' else -1))

dados.to_csv("tweets_analisados.csv", index=False)
print(dados)

plt.figure(figsize=(10, 6))
sns.boxplot(data=dados[["vader_pt", "vader_en", "bert_en", "bertimbau_pt"]])
plt.title("Comparação de Análises de Sentimento")
plt.xlabel("Métodos")
plt.ylabel("Score de Sentimento")
plt.savefig("comparacao_sentimento.png")
plt.show()

print("Atividade concluída! Dados salvos e gráficos gerados.")