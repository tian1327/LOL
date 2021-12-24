To run the ColBERT scripts on Colab:

1. Upload the data and scripts to google drive folder, and mount the google drive (optional)
2. change the %cd '/content/drive/MyDrive/CSCE 638 NLP Project/LOL_Data/' with the directory of the uploaded data folder
3. click run all

To run the text augmentation scripts on Colab:
1. Upload the data and scripts to google drive folder, and mount the google drive (optional)
2. change the %cd '/content/drive/MyDrive/CSCE 638 NLP Project/LOL_Data/' with the directory of the uploaded data folder
3. Due to limited google translation services, back translation has to be perfomed segment by segment on the text data. To do so, manually change the for loop starting indext in:
	for r in range(1978, df_train.shape[0]):
   
To run the .py scripts on local machine:
1. Change the data directory as instructed above
2. Install the dependencies as done in the .ipynb scripts
3. Run the .py scripts