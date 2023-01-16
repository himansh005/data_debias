from imports import *
from dataloader import *
from train import *
from data_transformer import *
from bias_finder  import *
from data_subsampler import *

def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__=="__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--working_dir', type=str)
  parser.add_argument('--config', type=str)
  parser.add_argument('--preprocess_only', action='store_true')

  config = {
        "name_mask_method":"smart_random_masking",
        "nonname_mask_method":"smart_random_masking",
        "naive_mask_token":"person",
        "seed":701,
        "persistence_dir":"working_dir",
        "save_data":True,
        "experiment_id":1,
        "load_saved_data":False,
        "epochs":1,
        "max_length":512,
        "mlm_prob":1.0,
        "model":"bert-base-uncased",
        "dataset_type":"remote",
        "dataset":"squad",
        "sample":10,
        "measure_initial_bias":True,
        "sample_method":"random",

  }
  args = parser.parse_args()

  #config = json.loads(args.config)
  root =  args.working_dir

  random_seed(config["seed"])
  
  wandb.login(key="ef63b680965922a125fe3650444a6697c61d9246") 
  
  run = wandb.init(
      name = "Experiment: "+ str(config["experiment_id"]),
      reinit = True, 
      project = "data_debias", 
      config = config
  )
  
  dataloader = CustomDataset(config, root, run)
  dataset = dataloader.load()

  if "measure_initial_bias" in config and config["measure_initial_bias"]:

    dataTransformer = BiasFinder(config, root, run)
    dataset = dataTransformer.modify_data(dataset)
  
  if "sample" in config:
    dataSubsampler = DataSubsampler(config, root, run, config["model"])
    dataset = dataSubsampler.subsample(dataset, config["sample"], config["sample_method"])

  dataTransformer = DataTransformer(config, root, run)
  dataset, new_words = dataTransformer.modify_data(dataset) #dataset, sample and intervention
  #print(dataset[0])
  if not args.preprocess_only:
    customTrainer = CustomTrainer(config, root, run)
    model_path = customTrainer.train(dataset, new_words)

  run.finish()
  print(model_path)
