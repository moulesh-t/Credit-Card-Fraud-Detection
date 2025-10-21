import os
import kaggle
# Downloading Kaggle dataset
def download():
    
    import os

def download():
    os.makedirs('./data', exist_ok=True)
    os.system('kaggle datasets download -d mlg-ulb/creditcardfraud -p ./data --unzip')
    file_path = './data/creditcard.csv'
    if os.path.exists(file_path):
        print("Dataset downloaded Successfully")
    else:
        print("Error in downloading dataset")


if __name__ == "__main__":
    download()