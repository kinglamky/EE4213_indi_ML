# EE4213_indi_ML
Set the correct path for the Amazon tsv file, this script is writen to process the amazon review as demo(https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset/data?select=amazon_reviews_us_Furniture_v1_00.tsv)
'amazon_reviews_us_Furniture_v1_00.tsv' was choosen for its small size and only read the first 100000 line of data (line68 to change the settign) for faster loading time

you can change the inputProductParent or inputProduct name to change the searched product, and change the SingleProductOverallAnalysis(text) to wanted item.
inputProductParent should be more stable and fast
Also, could change the number of keyTopic listed



This script utilize 4 difient classifier
1. the default one :distilbert-base-uncased-finetuned-sst-2-english
2. toixc-comment-model : martin-ha/toxic-comment-model
3. Emootion Cassiffier : michellejieli/emotion_text_classifier
4. Required Cassifier (stars) : LiYuan/amazon-review-sentiment-analysis

Pipline flow:
1. Read tsv file
2. analyze choosen product
3. filter out toxic comment by the toxic classifier
4. Run the remaining comment through above classifier after preprocessing, there is a function to handle text(>512 length) to do classifier
5. Utilize LDA and TF-IDF to extract most talk about keywords
6. The stars classifier was modify and output weighted satisfaction score base on the outputed stars from the classifier, and presented as a trend
7. Emotion Classifier output the emotion of the comment, and presented as a bar chart
8. output 6 kind of data: Total number of comment(show in terminal) , POS/NEG Distribution(graph), Top 5 mentioned keywords (show in terminal), POS/NEG comment with most helpful vote (show in terminal), weighted satisfaction score trend (graph), Emotion bar chart(graph)



Please install all listed lib for stable exicution, my venv is in this folder(https://connectpolyu-my.sharepoint.com/:f:/g/personal/a18134460a_connect_polyu_hk/EqJU-0GKYqZOkfPNhxjL08YBg--_urFa0CRTd2QdDf5zIQ?e=hoveSx)
