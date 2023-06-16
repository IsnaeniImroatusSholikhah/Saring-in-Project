import requests as req
from bs4 import BeautifulSoup as bs
from datetime import datetime
import csv
import gensim
from gensim import corpora
import tensorflow as tf
import transformers
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np

hades = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'}

def scrape_detik(hal,query):
    global hades
    a = 1
    toReturn = []
    for page in range(1,hal):
        url = f'https://www.detik.com/search/searchnews?query={query}&sortby=time&page={page}'
        ge = req.get(url,hades).text
        sop = bs(ge,'lxml')
        li = sop.find('div',class_='list media_rows list-berita')
        lin = li.find_all('article')
        for x in lin:
            link = x.find('a')['href']
            date = x.find('a').find('span',class_='date').text.replace('WIB','').replace('detikNews','').split(',')[1]
            headline = x.find('a').find('h2').text
            ge_ = req.get(link,hades).text
            sop_ = bs(ge_,'lxml')
            content = sop_.find_all('div',class_='detail__body-text itp_bodycontent')
            for x in content:
                x = x.find_all('p')
                y  = [y.text for y in x ]
                content_ = ''.join(y).replace('\n', '').replace('ADVERTISEMENT','').replace('SCROLL TO RESUME CONTENT','')
                # print(f'done[{a}] > {headline[0:40]}')
                toReturn.append(content_)
                a += 1
    return toReturn
                # with open('politik.csv','a')as file:
                #     wr = csv.writer(file, delimiter=',')
                #     wr.writerow([headline,date,link,content_])


def search_links(text, num_links):
    query = 'penjelasan mengenai ' + text + ' dari sumber yang terpercaya'
    search_url = 'https://www.google.com/search?q=' + query

    # Send a GET request to the search URL
    response = req.get(search_url)

    # Parse the HTML content using BeautifulSoup
    soup = bs(response.content, 'html.parser')

    # Find the search results links
    search_results = soup.find_all('a')

    # Extract the URLs from the search results
    links = []
    for link in search_results:
        url = link.get('href')
        if url.startswith('/url?q='):
            url = url[7:]  # Remove the '/url?q=' prefix
            if '&' in url:
                url = url[:url.index('&')]  # Remove additional parameters
            links.append(url)

    # Return the specified number of links or up to 5 links if available
    return links[:min(num_links, 10)]

def get_topic(text):
    # Preprocessing and preparing the text data
    news_data = [
        text
    ]

    # Tokenize the document
    tokenized_data = [news_data[0].split()]

    # Create a dictionary from the tokenized data
    dictionary = corpora.Dictionary(tokenized_data)

    # Convert tokenized data into a bag-of-words representation
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_data]

    # Perform LDA
    num_topics = 1  # Specify the number of topics
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    # Print the topics and their keywords
    num_words = 5  # Number of words to display per topic
    for idx, topic in lda_model.print_topics(num_topics=num_topics, num_words=num_words):
        print(f'Topic {idx + 1}: {topic}')
        print(topic)

    topic_keywords = topic

    # Extracting keywords
    keywords = [word.strip().split("*")[1].replace("\"", "") for word in topic_keywords.split("+")]

    link_recomendation = []

    def mySearch(myItem):
      query = 'penjelasan mengenai' + myItem + "dari sumber yang terpercaya"
      for j in search_links(query, num_links=2):
          link_recomendation.append(j) 
    
    for keyword in keywords:
      mySearch(keyword)

    return link_recomendation

max_length = 128  # Maximum length of input sentence to the model.
batch_size = 32
epochs = 10

#mendefinisikan label yang ada , 0=entailment, 1 = netral, dan 2= kontradiksi
# Labels in our dataset.
labels = ["entailment", "neutral", "contradiction"]


class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "indobenchmark/indobert-base-p2", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()
        
    def __cleaning(self, text:str):
        # clear punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))

        # clear multiple spaces
        text = re.sub(r'/s+', ' ', text).strip()

        return text

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
            truncation = True,
            truncation_strategy='only_first'
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)



loaded_model = tf.keras.models.load_model('./model.h5', custom_objects={"TFBertModel": transformers.TFBertModel})

def check_similarity(sentence1, sentence2):
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )

    proba = loaded_model.predict(test_data[0])[0]
    idx = np.argmax(proba)
    final_proba = int(proba[idx] * 100)
    proba = "{:.0f}%".format(final_proba)
    pred = labels[idx]
    # labels = ["entailment", "neutral", "contradiction"]
    if(pred == 'entailment'):
      return 'Berita yang anda masukan '+ proba +' memiliki kesamaan dengan sumber terpercaya'
    elif(pred == 'contradiction'):
      return 'Berita yang anda masukan '+ proba +' memiliki perbedaan dengan sumber terpercaya'
    elif(pred == 'neutral'):
      return 'Berita yang anda masukan '+ proba +' tidak memiliki keterkaitan dengan sumber terpercaya'

app = Flask(__name__)

@app.route('/get-source-of-truth', methods=['POST'])
def getSourceOfTruth():
    data = request.form
    title = data['title']

    if title:
        source_of_truth = scrape_detik(3, title)
        print(source_of_truth)
        return jsonify({'news_source_of_truth': source_of_truth})
    else:
        return jsonify({'error': 'Data not provided'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    source_of_truth = data['source_of_truth']
    news_text = data['news_text']

    if source_of_truth:
        result_comparing = check_similarity(news_text, source_of_truth)
        return jsonify({'predicition_result': result_comparing})
    else:
        return jsonify({'error': 'Data not provided'})

@app.route('/get-recomendation', methods=['POST'])
def getRecomendation():
    data = request.form
    news_text = data['news_text']

    if news_text:
        article_recomendation = get_topic(news_text)
        return jsonify({'recomendations': article_recomendation})
    else:
        return jsonify({'error': 'Data not provided'})

if __name__ == '__main__':
    app.run()