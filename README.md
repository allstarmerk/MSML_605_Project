# MSML_605_Project
MSML/MSAi project. ITs a facial recognition system similar to faceId

-Group: Johnathan Sheikh, Renae Ricketts

-Main branch is for after bugs are fixed in working branch. Then we merge the working branch into the main branch. This allows for the main branch to be bug free and always in working condition. Then the end product will be the main branch. Working branch is essentialy the development branch and main being the production branch. 

-After you have work completed on the working branch you create a pull request. If u wanna use the UI of github you go to pull request. Click on create new pull request. Then you select the working branch and the main branch. Then you click on create pull request. Then you can add a description of the changes you made. Then you can click on create pull request again. Then the pull request will be created and you can see it in the pull request tab. Then you can merge the pull request if there are no conflicts. If there are conflicts then you have to resolve them before merging.

How to run: 

1. Set up the env
    -git clone <repo url>
    -cd MSML_605_Project

    --> python -m venv .venv   or  for windows you may have to use -->  py -m venv .venv
      -  .venv\Scripts\Activate

    - pip install -r requirements.txt


    2. Ingest LFW dataset and generate pairs:
      run:   python scripts/lfw_ingestion_script.py
      -this should download lfw data set(170MG) and saves/creates data/manifest.json & data/pairs/
    
    3. Run similarity python script
       - python scripts/benchmark_similarity.py 




    4. To verify the pair splits Run
      run ---> python -c "import numpy as np; pairs = np.load('data/pairs/train_pairs.npy'); labels = np.load('data/pairs/train_labels.npy'); print('pairs shape:', pairs.shape); print('positive pairs:', labels.sum()); print('negative pairs:', (labels==0).sum())"


      Expected Output: Expected output:
        pairs shape: (3000, 2)
        positive pairs: 1500
        negative pairs: 1500



    5. To verify determinism (run ingestion script a 2nd time)
        - python scripts/lfw_ingestion_script.py
          
          Then Check that data/manifest.json has identical counts to the first run. to comfirm determinism
       
        Design Choices:
    - We use a hard coded seed in config/ to meet the determinisc requirment. Seed is fixed numpy, tensorflow, and python random.

    - saved pair indices saved as a .npy so evaluation is always accurate and reproducable

    - Most of the settings needed are hard coded in /configs/dataset.yaml



    - To test the logic and math for Similarity.py you can run benchmark_similarity.py
    
              run-->     - python scripts/benchmark_similarity.py

        - This will provide a output showing the comparision of using the loop vs Numpy for both cosine and euclidean calculations. As well as having a unit test at the end checking the math of the math functions.




