# EE4213_indi_ML
Set the correct path for the Amazon tsv file, this script is writen to process the amazon review as demo(https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset/data?select=amazon_reviews_us_Furniture_v1_00.tsv)
you can change the inputProductParent or inputProduct name to change the searched product, and change the SingleProductOverallAnalysis(text) to wanted item

This script utilize 4 difient classifier
1. the default one :distilbert-base-uncased-finetuned-sst-2-english
2. toixc-comment-model : martin-ha/toxic-comment-model
3. Emootion Cassiffier : michellejieli/emotion_text_classifier
4. Required Cassifier : LiYuan/amazon-review-sentiment-analysis


Please install all listed lib for stable exicution, my venv is in this folder(https://connectpolyu-my.sharepoint.com/:f:/g/personal/a18134460a_connect_polyu_hk/EqJU-0GKYqZOkfPNhxjL08YBg--_urFa0CRTd2QdDf5zIQ?e=hoveSx)
