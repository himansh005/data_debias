from imports import *
from transformers import pipeline
class BiasFinder:

  def __init__(self, config: dict, root: str = ".", wandb=None, model="bert-base-uncased"):

    self.female_names = set()
    self.male_names = set()
    
    self.m2f = {}
    self.f2m = {}
    self.config = config
    self.wandb = wandb

    self.scores = []
    self.words_scanned=0
    self.rows_changed=0
    self.swaps = []
    self.new_words = []

    self.mask_filler = pipeline(
      "fill-mask", model=model
    )

    self.any2neutral = defaultdict(list)

    self.root = root
    self.random = Random(config["seed"])

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    self.logger = logging.getLogger(__name__)

    # 1. Intialize models and utils
    self.stemmer = SnowballStemmer("english")

    # 2. Populate non-names
    with open(os.path.join(root, "data", "male_female_pairs.txt"), "r") as f:
      
      data = f.readlines()
      for i in range(len(data)):
        
        if "=>" in data[i]:
          terms = data[i].split("=>")
        else:
          terms = data[i].split("=")
        
        if(len(terms)==1):
          continue

        male_word = terms[0].strip().lower()
        female_word = terms[1].strip().lower()

        self.m2f[male_word] = female_word
        self.m2f[self.stemmer.stem(male_word)] = female_word

        self.f2m[female_word] = male_word
        self.f2m[self.stemmer.stem(female_word)] = male_word

    with open(os.path.join(root, "data", "cda_default_pairs.json"), "r") as f:
      data = json.loads(f.read())
      for i in data:
        self.f2m[i[1]] = i[0]
        self.m2f[i[0]] = i[1]

    # 3. Populate names
    with open(os.path.join(root, "data","female_names.txt"), "r") as f:
      self.female_names = set([x.strip().lower() for x in f.readlines()])
    
    with open(os.path.join(root, "data", "male_names.txt"), "r") as f:
      self.male_names = set([x.strip().lower() for x in f.readlines()])
    
    with open(os.path.join(root, "data", "names_pairs_1000_scaled.json"), "r") as f:
      
      data = json.loads(f.read())
      for i in data:
        self.male_names.add(i[0])
        self.female_names.add(i[1])

    # 4. Populate neutrals
    with open(os.path.join(root, "data", "search-and-suggest.json"), "r") as f:
      data = json.loads(f.read())

    for obj in data:
      for item in obj["search"]:
        self.any2neutral[item]+=obj["suggest"]
        self.any2neutral[self.stemmer.stem(item)]+=obj["suggest"]
    
    for key in self.any2neutral:
      self.any2neutral[key] = list(set(self.any2neutral[key]))

  def scan(self, text):
    '''
      text: an incoming sentence/paragraph.
    '''
    self.words_scanned = 0
    self.rows_changed = 0
    self.swaps = []
    self.opp_words = []
    text = text.lower()
    original_text = copy.deepcopy(text)
    
    text = word_tokenize(text)
    for i in range(len(text)):

      # HOOK
      if text[i] in self.female_names:
        text[i] = "[MASK]"
        self.words_scanned+=1
        continue
      
      # HOOK
      if text[i] in self.male_names:
        text[i] = "[MASK]"
        self.words_scanned+=1
        continue
      
      text_stems = {}

      #stem the text if not present in non-names
      if text[i] not in self.f2m and text[i] not in self.m2f:
        if text[i] not in text_stems:
          text_stem = self.stemmer.stem(text[i])
          text_stems[text[i]] = text_stem
        else:
          text[i] = text_stems[text[i]] 
      
      # HOOK
      if text[i] in self.f2m:
        text[i] = "[MASK]"
        self.words_scanned+=1
        continue
      
      # HOOK
      if text[i] in self.m2f:
        text[i] = "[MASK]"
        self.words_scanned+=1
        continue
    
    text = " ".join(text)
    self.logger.debug(text)

    if self.words_scanned==0:
      self.scores.append(-1)
      return text
    fill_stems = {}

    predictions = self.mask_filler(text)
    score = 0.0

    for prediction in predictions:
      
      other_map = {}
      if len(prediction) == 0:
        continue
      if type(prediction) is not list:
        prediction = [prediction]

      for candidate in prediction:

        possible_fill = candidate["token_str"]
        possible_fill_score = candidate["score"]

        other_possible = [x["token_str"] for x in prediction if x["token_str"]!=possible_fill]
        other_possible_scores = [x["score"] for x in prediction if x["token_str"]!=possible_fill]
        other_possible_stems = []

        if possible_fill not in self.f2m and possible_fill not in self.m2f:
          if possible_fill not in fill_stems:
            possible_fill_stem = self.stemmer.stem(possible_fill)
            fill_stems[possible_fill] = possible_fill_stem
            possible_fill = possible_fill_stem
          else:
            possible_fill = fill_stems[possible_fill]

        for i in range(len(other_possible)):

          if other_possible[i] not in other_map:
              possible_fill_stem = self.stemmer.stem(other_possible[i])
              other_map[other_possible[i]] = possible_fill_stem
              other_possible_stems.append(possible_fill_stem)
          else:
              other_possible_stems.append(other_map[other_possible[i]]) 

        if possible_fill in self.female_names:
          for i in range(len(other_possible)):
            self.logger.debug("{} ==> {}".format(possible_fill, other_possible[i]))
            if other_possible[i] in self.male_names:
              conf_diff = abs(other_possible_scores[i]-possible_fill_score)
              self.opp_words.append((other_possible[i], possible_fill, conf_diff))
              score+=conf_diff
        
        elif possible_fill in self.male_names:
          for i in range(len(other_possible)):
            self.logger.debug("{} ==> {}".format(possible_fill, other_possible[i]))
            if other_possible[i] in self.female_names:
              conf_diff = abs(other_possible_scores[i]-possible_fill_score)
              self.opp_words.append((other_possible[i], possible_fill, conf_diff))
              score+=conf_diff
        
        elif possible_fill in self.f2m:
          for i in range(len(other_possible)):
            self.logger.debug("{} ==> {}".format(possible_fill, other_possible[i]))
            if other_possible[i]==self.f2m[possible_fill] or other_possible_stems[i]==self.f2m[possible_fill]:
              conf_diff = abs(other_possible_scores[i]-possible_fill_score)
              self.opp_words.append((other_possible[i], possible_fill, conf_diff))
              score+=conf_diff
        
        elif possible_fill in self.m2f:
          for i in range(len(other_possible)):
            self.logger.debug("{} ==> {}".format(possible_fill, other_possible[i]))
            if other_possible[i]==self.m2f[possible_fill] or other_possible_stems[i]==self.m2f[possible_fill]:
              conf_diff = abs(other_possible_scores[i]-possible_fill_score)
              self.opp_words.append((other_possible[i], possible_fill, conf_diff))
              score+=conf_diff

    self.scores.append(score)
    if(original_text!=text):
      self.rows_changed+=1

    return text

  def masking_actions(self, text, word_type=None):
    
    gender_word = text
    
    if word_type=="female_name":
      opp_gender_word = list(self.female_names)[self.random.randint(0, len(self.female_names))]
    
    elif word_type=="male_name":
      opp_gender_word = list(self.male_names)[self.random.randint(0, len(self.male_names))]

    elif word_type=="female_nonname":
      opp_gender_word = self.f2m[gender_word]
    
    elif word_type=="male_nonname":
      opp_gender_word = self.m2f[gender_word]
  
    text = "[MASK]"           
  
    self.logger.debug("{} => {}".format(gender_word, text))
    self.swaps.append((gender_word, text))
    self.new_words.append(text)
    return text, opp_gender_word

  def process_batch(self, batch_data, fields_to_mask):
    for feild in fields_to_mask:
      for i in range(len(batch_data[feild])):
        batch_data[feild][i] = self.scan(text=batch_data[feild][i])
    return batch_data

  def modify_data(self, dataset, subset=None):
    

    original_dataset = copy.deepcopy(dataset)
    if "load_saved_data" in self.config and self.config["load_saved_data"]:
      data_output_path = os.path.join(self.root, self.config["persistence_dir"], "data", str(self.config["experiment_id"]), str(self.config["dataset"]+"_"+self.config["model"]+"_confidence.h5"))
      if(os.path.exists(data_output_path)):
        dataset = dataset.load_from_disk(data_output_path)
        self.logger.warning("Reusing data from {}".format(data_output_path))
        return dataset
        
    args = {
              "fields_to_mask":["text"]
          }

    dataset = dataset.map(self.process_batch, fn_kwargs=args, batched=True)
    dataset = original_dataset.add_column("bias", self.scores)

    output_path = os.path.join(self.root, self.config["persistence_dir"], "intervention_stats", str(self.config["experiment_id"]), "opposite_words_confidences.csv")  
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data=self.opp_words, columns=["word", "opp_word", "confidence_difference"])
    df.to_csv(output_path, index=False)

    self.words_scanned=0
    self.rows_changed=0
    self.swaps = []
    
    self.new_words = []
    self.wandb.log({"avg_dataset_bias":sum(self.scores)/len(self.scores)})
    self.logger.debug("Average Prediction Bias: {}".format(str(sum(self.scores)/len(self.scores))))

    data_output_path = os.path.join(self.root, self.config["persistence_dir"], "data", str(self.config["experiment_id"]), str(self.config["dataset"]+"_"+self.config["model"]+"_confidence.h5"))
    if self.config["save_data"]:
        Path(os.path.dirname(data_output_path)).mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(data_output_path)

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
#         "load_saved_data":False
#       }
#     ]
    

#   for config in experiments:

#     dataTransformer = BiasFinder(config, "drive/Othercomputers/mba/")
#     dataset = load_dataset("csv", data_files="ssd.csv", split="train")
#     dataset = dataset.rename_column("context", "text")
#     dataset = dataTransformer.modify_data(dataset)
#     print(dataset)
  
