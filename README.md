# MSML_605_Project
MSML/MSAI project. It's a facial recognition system similar to faceId

- Group: Johnathan Sheikh, Renae Ricketts

- Main branch is for after bugs are fixed in working branch. Then we merge the working branch into the main branch. This allows for the main branch to be bug free and always in working condition. Then the end product will be the main branch. Working branch is essentialy the development branch and main being the production branch. 

- After you have work completed on the working branch you create a pull request. If u wanna use the UI of github you go to pull request. Click on create new pull request. Then you select the working branch and the main branch. Then you click on create pull request. Then you can add a description of the changes you made. Then you can click on create pull request again. Then the pull request will be created and you can see it in the pull request tab. Then you can merge the pull request if there are no conflicts. If there are conflicts then you have to resolve them before merging.

How to run: 

1. To set up the environment, run the following commands:
```
git clone <repo url>
cd MSML_605_Project
python -m venv .venv   (or for Windows you may have to use -->  py -m venv .venv)
.venv\Scripts\Activate
pip install -r requirements.txt
```
<br/>

2. Ingest LFW dataset and generate pairs, run the following:
```
python scripts/lfw_ingestion_script.py
```
- This should download lfw data set(170MG) and saves/creates data/manifest.json & data/pairs/
<br/>


3. To verify the pair splits, run the following:
```
python -c "import numpy as np; pairs = np.load('data/pairs/train_pairs.npy'); labels = np.load('data/pairs/train_labels.npy'); print('pairs shape:', pairs.shape); print('positive pairs:', labels.sum()); print('negative pairs:', (labels==0).sum())"
```

Expected Output:
  - pairs shape: (3000, 2)
  - positive pairs: 1500
  - negative pairs: 1500
<br/>


4. To verify determinism, run ingestion script a 2nd time:
```
python scripts/lfw_ingestion_script.py
```    
  
- Then compare new data/manifest.json with the 1st run's manifest summary printed on the command terminal and check if they have identical counts. to comfirm determinism
<br/> 


5. To run the Similarity Benchmark test, run the following:
```
python scripts/benchmark_similarity.py
```
  
- This will provide a output showing the comparision of using the loop vs Numpy for both cosine and euclidean calculations. It also has a unit test at the end checking the math of the math functions.
<br/>

Design Choices:
  - We use a hard coded seed in config/ to meet the determinisc requirment. Seed is fixed numpy, tensorflow, and python random.

  - Saved pair indices are saved as a .npy so evaluation is always accurate and reproducable

  - Most of the settings needed are hard coded in /configs/dataset.yaml




