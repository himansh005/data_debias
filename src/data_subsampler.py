from imports import *
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel

class DataSubsampler:

  def __init__(self, config: dict, root: str = ".", wandb=None, model="bert-base-uncased"):

    self.config = config
    self.root = root

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    self.logger = logging.getLogger(__name__)

    self.tokenizer = AutoTokenizer.from_pretrained(model)
    self.model = AutoModel.from_pretrained(model, output_hidden_states=True)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model = self.model.to(self.device)
    self.model.eval()
  
  def subsample(self, dataset, k):

    tokenized_dataset = dataset.map(lambda x: self.tokenizer(x['text'],max_length=self.config["max_length"], truncation=True), batched=True)
    dataset = pd.DataFrame(tokenized_dataset)
    dataset.drop_duplicates(subset="text", inplace=True)
    dataset.reset_index(drop=True)
    embeddings = []

    for i in tqdm(range(len(tokenized_dataset))):

      text_ids = tokenized_dataset[i]["input_ids"]

      text_ids = torch.LongTensor(text_ids)
      text_ids = text_ids.unsqueeze(0)
      text_ids = text_ids.to(self.device)
      with torch.no_grad():
        out = self.model(input_ids=text_ids)

      hidden_states = out[2]
      last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]

      emb = torch.cat(tuple(last_four_layers), dim=-1)
      emb = torch.mean(emb, dim=1).squeeze()
      embeddings.append(emb.numpy())
    
    data = np.array(embeddings)
    kmeans = KMeans(n_clusters=k, random_state=self.config["seed"]).fit(data)
    
    closest_pt_idx = []
    for iclust in range(kmeans.n_clusters):

        #TODO: Fix segmentation fault bug
        cluster_pts_indices = np.where(kmeans.labels_ == iclust)[0]

        cluster_cen = kmeans.cluster_centers_[iclust]
        xx = [euclidean(data[idx], cluster_cen) for idx in cluster_pts_indices]
        min_idx = np.argmin(xx)
        closest_pt_idx.append(cluster_pts_indices[min_idx])

    dataset = dataset[dataset.index.isin(closest_pt_idx)]
    dataset = dataset.reset_index(drop=True)
    dataset = dataset.drop(columns=["input_ids", "token_type_ids", "attention_mask"])

    return Dataset.from_pandas(dataset)


  def subsample(self, dataset, samples, method="random"):

    methods = ["random", "most-biased", "diverse", "diverse-most-biased"]
    if method not in methods:
      raise Exception("Method can only be one of: {}".format(",".join(methods)))
    
    if type(samples) is str:
      samples = int(len(dataset)*(int(samples[:-1])/100))
    
    self.logger.debug("Using {} out of {} rows in the dataset".format(samples, len(dataset)))
    
    if method=="random":
      dataset = dataset.shuffle(seed=self.config["seed"]).select(range(samples))
    
    elif method=="most-biased":
      dataset = dataset.sort('bias', reverse=True)
      dataset = dataset.select(range(samples))

    
    #TODO: add other two methods

    return dataset

# if __name__=="__main__":

#   experiments = [
#       {
#         "name_mask_method":"smart_random_masking",
#         "nonname_mask_method":"smart_random_masking",
#         "naive_mask_token":"person",
#         "seed":701,
#         "persistence_dir":"src/logs",
#         "save_data":True,
#         "experiment_id":1,
#         "load_saved_data":False,
#         "max_length":512
#       }
#     ]
    

#   for config in experiments:
#     print(config)
#     dataTransformer = DataSubsampler(config, "/Users/Himanshu/Developer/")
#     dataset = load_dataset("csv", data_files="ssd.csv", split="train")
#     # dataset = dataset.rename_column("context", "text")
#     dataset = dataset.shuffle(seed=701).select(range(20))
#     dataset = dataTransformer.subsample(dataset, 10)
#     print(dataset)
  
