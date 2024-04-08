import os

from dataloader import Dataset
from evaluate import Tester 
import config.config as config

def main():
    path = '../data/{}/meta'.format(config.data_folder)
    if config.regenerate_graphdata:
        # remove path/processed folder
        import shutil
        if os.path.exists(path + "/processed"):
            print("remove path/processed folder")
            shutil.rmtree(path + "/processed") 
    data = Dataset(path)

    tester = Tester(path, data) 
    print("Start Testing...")
    # load model
    tester.load_model(config.model_path)
    tester.evaluate(0, evaltask=config.task)
        

if __name__ == '__main__':
    main()