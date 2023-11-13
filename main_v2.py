import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
import string
from collections import defaultdict
from transformers import pipeline, TextClassificationPipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import RegexpTokenizer, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from bs4 import BeautifulSoup
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings


#Read less row to run faster
debugMode = False

#File Path it could be any tsv file
#File Path it could be any tsv file, i choose the smallest file for fast loading
tsvFileList = ['amazon_reviews_us_Mobile_Electronics_v1_00.tsv',
        'amazon_reviews_multilingual_US_v1_00.tsv',
        'amazon_reviews_us_Baby_v1_00.tsv',
        'amazon_reviews_us_Digital_Software_v1_00.tsv',
        'amazon_reviews_us_Furniture_v1_00.tsv',
        'amazon_reviews_us_Major_Appliances_v1_00.tsv'
        ]
FilePath = "AmazonReviewDataset/"+tsvFileList[4]

# initial value
model_path_star = "LiYuan/amazon-review-sentiment-analysis"
model_path_hatespeech = "martin-ha/toxic-comment-model"

tokenizer_star = AutoTokenizer.from_pretrained(model_path_star)
model_star = AutoModelForSequenceClassification.from_pretrained(model_path_star)
tokenizer_hatespeech = AutoTokenizer.from_pretrained(model_path_hatespeech)
model_hatespeech = AutoModelForSequenceClassification.from_pretrained(model_path_hatespeech)



classifier = pipeline("sentiment-analysis")
classifier_LiYuan = pipeline("sentiment-analysis", model= model_star,tokenizer=tokenizer_star)
classifier_hate = TextClassificationPipeline(model=model_hatespeech, tokenizer=tokenizer_hatespeech)
classifier_emotion = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
#nltk.download('stopwords')
#nltk.download('punkt')
#print(classifier_hate('small eyes china boi'))

# Input Desired file

def showProgress(word):
    print("\n====================")
    print("[" + word + "]")


# Read File
def read_tsv_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        headers = next(reader)  # Get the first row as headers
        for i, row in enumerate(reader):
            if debugMode and i == 2000:
                break
            elif i == 100000:
                break
            #data.append(dict(zip(headers, filter_text(row[0]))))
            #remove HTML tag
            #row[13] = BeautifulSoup(row[13], "html.parser")
            data.append(dict(zip(headers, row)))
    return data

def filter_text(text):
    #soup = BeautifulSoup(text, "html.parser")
    #text = soup.get_text()

    # prep the input
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text.replace('\n', ' '))
    pos_words = pos_tag(words)

    # Remove list
    stop_words = set(stopwords.words('english')) 
    pos_list = ['PRP', 'PRP$','UH','WDT','TO']

    
    filtered_words = [word for word, pos in pos_words if 
                      word.isalpha() 
                      and word not in stop_words 
                      and pos not in pos_list
                      ]

    return " ".join(filtered_words)

def summarize_text(text):
    # Tokenize the sentences
    sentences = sent_tokenize(text)

    # Generate the frequency table for all words
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word not in stopWords:
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1

    # Calculate the score for each sentence
    sentence_scores = dict()
    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentence_scores:
                    sentence_scores[sentence] += freq
                else:
                    sentence_scores[sentence] = freq

    # Get the average score for the sentences
    average = sum(sentence_scores.values()) / len(sentence_scores)

    # Generate the summary
    summary = ''
    for sentence in sentences:
        if sentence in sentence_scores and sentence_scores[sentence] > (1.2 * average):
            summary += " " + sentence
    return summary


def split_text(text, chunk_length):
    chunks = [text[i:i+chunk_length] for i in range(0, len(text), chunk_length)]
    return chunks

def classifierForLongReview(sentiments,mode):
    chunk_length = 512
    chunks = split_text(sentiments, chunk_length)
    if mode =="normal":
        sentiments = [classifier(chunk) for chunk in chunks]
    elif mode == "hate":
        sentiments = [classifier_hate(chunk) for chunk in chunks]
    label_counts = {}
    label_scores = {}

    for sentiment in sentiments:
        label = sentiment[0]['label']
        score = sentiment[0]['score']
        if label in label_counts:
            label_counts[label] += 1
            label_scores[label] += score
        else:
            label_counts[label] = 1
            label_scores[label] = score

    majority_label = max(label_counts, key=label_counts.get)
    majority_score = label_scores[majority_label] / label_counts[majority_label]

    return [{'label': majority_label, 'score': majority_score}]

#analyze the reivew
def analyzeReivew(text):
    #classifier(review["review_body"][0:511])
    text = filter_text(text)
    #print(text)
    if len(text) > 512:
        return classifierForLongReview(text,"normal")
    else:
        return classifier(text)

dataSet = read_tsv_file(FilePath)

def AnalyzeHateSpeech(text):
    #text = filter_text(text)
    #print(text)
    if len(text) > 512:
        return classifierForLongReview(text,"hate")
    else:
        return classifier_hate(text)

#single product overall Analysis
def SingleProductOverallAnalysis(product):
    cnt_pos, cnt_neg = 0,0
    dataSet_indi, dataSet_star =[],[]
    Max_helpful_POS, Max_helpful_NEG = 0,0
    Max_helpful_POS_cm, Max_helpful_NEG_cm = " "," "
    preprocessed_documents = []
    dataSet_Emo = {
        'joy':0,
        'neutral':0,
        'surprise':0,
        'anger':0,
        'disgust':0,
        'sadness':0, 
        'fear':0
        }
    bottom_text = "Top 5 Keyboards:\n"
    for review in dataSet:
        
        if review["product_title"] == product or review["product_parent"] == product:

            

            

        #Hate Speech Ana, dont proccess the review if the review is hate speech
            if AnalyzeHateSpeech(review["review_body"])[0]["label"] == 'non-toxic':


                


                # Text preprocessing
                tokenizer = RegexpTokenizer(r'\w+')
                words = tokenizer.tokenize(review["review_body"].lower().replace('\n', ' '))
                pos_words = pos_tag(words)
                stop_words = set(stopwords.words('english'))
                pos_list = ['PRP', 'PRP$','UH','WDT','TO','WDT', 'VB','VBG','RB']
                want_list = ['NN','NNS','RBR']
                filtered_tokens = [token for token, pos in pos_words if 
                                   token.isalpha() 
                                   and token not in stop_words
                                   and pos not in pos_list
                                   and pos in want_list
                                   ]
                preprocessed_documents.append(filtered_tokens)

                

            #Pos/Neg Ana
                result_indi = analyzeReivew(review["review_body"])
                if result_indi[0]['label'] == "POSITIVE":
                    cnt_pos += 1
                    if int(review["helpful_votes"]) > int(Max_helpful_POS):
                        Max_helpful_POS = int(review["helpful_votes"])
                        Max_helpful_POS_cm = review["review_body"]
                else: 
                    cnt_neg += 1
                    if int(review["helpful_votes"]) > int(Max_helpful_NEG):
                        Max_helpful_NEG = int(review["helpful_votes"])
                        Max_helpful_NEG_cm = review["review_body"]

            
            #print(review["review_body"],"\n",result_indi[0]['label'],"\n",Max_helpful_POS, "\n",Max_helpful_NEG,"\n","\n","\n")
            


            #star Ana
                Analys_star = classifier_LiYuan(filter_text(review["review_body"][0:511]))[0]
                
                #weighting the star base on score the lower the score the closer it get to avg(3)
                if int(Analys_star['label'][0]) >=3:
                    star_weighted =3+(float(Analys_star['label'][0])-3) * (Analys_star['score'])
                else: 
                    star_weighted =3-(3-float(Analys_star['label'][0])) * (Analys_star['score'])
                dataSet_indi.append({
                    #'star_og':Analys_star['label'][0],
                    #'score':Analys_star['score'],
                    'star_weighted':star_weighted,
                    'review_date':review["review_date"]
                    })
            #Emo Ana
                Analys_Emo = classifier_emotion(filter_text(review["review_body"])[0:511])
                dataSet_Emo[Analys_Emo[0]['label']] += Analys_Emo[0]['score']


            
    fig, (pie, bar, scatter) = plt.subplots(1,3)

    #Print out Ana result
    cnt_all = cnt_pos+cnt_neg
    percantage_POS = cnt_pos/(cnt_all)
    percantage_NEG = cnt_neg/(cnt_all)
    print("Total comment:",cnt_all,"\nPOS:",percantage_POS , "\nNEG:",percantage_NEG)

    
    pie_content =[int(cnt_pos),int(cnt_neg)]
    pie_label='POS','NEG'
    pie.pie(
        pie_content,
        labels=pie_label,
        autopct='%1.1f%%'
        
        )
    pie.set_title('POS/NEG Distribution')



    #print out Emo stat as bar chart
    #print(dataSet_Emo)
    labels = list(dataSet_Emo.keys())
    scores = list(dataSet_Emo.values())

    # Create the bar chart
    bar.bar(labels, scores)

    # Add labels and title
    bar.set_xlabel('Label')
    bar.set_ylabel('Score')
    bar.set_title('Emo Distribution Chart')

    



    # Topic modeling using LDA
    dictionary = corpora.Dictionary(preprocessed_documents)
    corpus = [dictionary.doc2bow(doc) for doc in preprocessed_documents]

    lda_model = models.LdaModel(corpus, num_topics=1, id2word=dictionary, passes=10)

    # Extracting keywords using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(doc) for doc in preprocessed_documents])

    # Accessing feature names indirectly through vocabulary_
    feature_names = tfidf_vectorizer.vocabulary_
    feature_names = {i: feature for feature, i in feature_names.items()}
    sorted_feature_names = [feature_names[i] for i in sorted(feature_names.keys())]

    keywords_per_topic = []

    for topic in lda_model.get_topics():
        top_keywords_idx = topic.argsort()[-5:][::-1]  # Get top 5 keywords for each topic
        top_keywords = [sorted_feature_names[idx] for idx in top_keywords_idx]
        keywords_per_topic.append(top_keywords)

    # Printing the extracted topics and keywords
    
    for i, topic_keywords in enumerate(keywords_per_topic):
        print(f"Topic {i+1}:")
        print(", ".join(topic_keywords))
        #bottom_text += f"{topic_keywords}   "
        print()
    
    
        

    #printing the longest comment for POS/NEG
    
    
    if not debugMode:
        print("Most helpful POS review: ",
              BeautifulSoup(summarize_text(Max_helpful_POS_cm), "html.parser").get_text(),
              "\n\nMost helpful NEG review: ",
              BeautifulSoup(summarize_text(Max_helpful_NEG_cm), "html.parser").get_text()
              )
    
    #Sentiment Trend Analysis base on time
    dataSet_indi = sorted(dataSet_indi, key = lambda x: x['review_date'])
   
    print("\nstar trend:")
    '''
    for temp in dataSet_indi:
        #print(temp)
        print(temp['review_date'], "    ", temp['star_weighted'])
    '''
    
    graph_dates = [d["review_date"] for d in dataSet_indi]
    date_datetime = pd.to_datetime(graph_dates)
    graph_scores = [d["star_weighted"] for d in dataSet_indi] 
    date_numeric = list(range(len(graph_dates)))
    
    def index_formula(number):
        index = int(number / 10)
        if index < 1:
            return 1
        else:
            return index
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Polyfit may be poorly conditioned')
        sns.regplot(
            x=date_numeric, 
            y=graph_scores, 
            scatter=True, 
            label='Scores', 
            #logistic=True,
            order = index_formula(cnt_all),
            line_kws={'color':'darkorange'}, 
            scatter_kws={'color':'pink'}
            )

    # Customize the graph
    scatter.set_xlabel('Date')
    scatter.set_ylabel('Score')
    scatter.set_title('Score Trend')
    #plt.xticks(date_numeric, date_datetime.dt.strftime('%Y-%m-%d'), rotation=45)
    # Show the graph
    """plt.text(0.5, -0.2, bottom_text, ha='center', va='center', transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.3)"""

    plt.show()
    
    #dataSet_star.append(classifier_LiYuan())
            #print(review["review_body"])
            #print(analyzeReivew(review["review_body"]))


    


# testing



def main():
    
    #dataSet.sort(key='product_id')
    #print("*** Testing Return ***")
    print("Input:")
    #input = input()
    inputProduct = 'Sleep Innovations Shiloh Memory Foam Mattress'
    inputProductParent ='880806683'
    print(inputProductParent)
    SingleProductOverallAnalysis(inputProductParent)

    






main()
    

