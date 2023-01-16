from imports import *

class DataTransformer:

  def __init__(self, config: dict, root: str = ".", wandb=None):

    self.female_names = set()
    self.male_names = set()
    
    self.m2f = {}
    self.f2m = {}
    self.config = config
    self.wandb = wandb

    self.words_scanned=0
    self.rows_changed=0
    self.swaps = []
    self.new_words = []

    self.any2neutral = defaultdict(list)

    self.root = root
    self.random = Random(config["seed"])

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    self.logger = logging.getLogger(__name__)

    # 1. Intialize models and utils
    self.stemmer = SnowballStemmer("english")

    # 2. Populate non-names
    with open(os.path.join(root, "data_debias/data/", "male_female_pairs.txt"), "r") as f:
      
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

    with open(os.path.join(root, "data_debias/data/", "cda_default_pairs.json"), "r") as f:
      data = json.loads(f.read())
      for i in data:
        self.f2m[i[1]] = i[0]
        self.m2f[i[0]] = i[1]

    # 3. Populate names
    with open(os.path.join(root, "data_debias/data/","female_names.txt"), "r") as f:
      self.female_names = set([x.strip().lower() for x in f.readlines()])
    
    with open(os.path.join(root, "data_debias/data/", "male_names.txt"), "r") as f:
      self.male_names = set([x.strip().lower() for x in f.readlines()])
    
    with open(os.path.join(root, "data_debias/data/", "names_pairs_1000_scaled.json"), "r") as f:
      
      data = json.loads(f.read())
      for i in data:
        self.male_names.add(i[0])
        self.female_names.add(i[1])

    # 4. Populate neutrals
    with open(os.path.join(root, "data_debias/data/", "search-and-suggest.json"), "r") as f:
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
    text = text.lower()
    original_text = copy.deepcopy(text)
    
    text = word_tokenize(text)
    for i in range(len(text)):

      # HOOK
      if text[i] in self.female_names:
        text[i] = self.masking_actions(text[i], word_type="female_name", method=self.config["name_mask_method"])
        self.words_scanned+=1
        continue
      
      # HOOK
      if text[i] in self.male_names:
        text[i] = self.masking_actions(text[i], word_type="male_name", method=self.config["name_mask_method"])
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
        text[i] = self.masking_actions(text[i], word_type="female_nonname", method=self.config["nonname_mask_method"])
        self.words_scanned+=1
        continue
      
      # HOOK
      if text[i] in self.m2f:
        text[i] = self.masking_actions(text[i], word_type="male_nonname", method=self.config["nonname_mask_method"])
        self.words_scanned+=1
        continue
      
      # custom HOOK
      if self.config["nonname_mask_method"]=="neutral_masking" or self.config["name_mask_method"]=="neutral_masking":
        if text[i] in self.any2neutral:
          text[i] = self.masking_actions(text[i], word_type=None, method=self.config["nonname_mask_method"])
          self.words_scanned+=1
          continue 

    text = " ".join(text)
    if(original_text!=text):
      self.rows_changed+=1

    return text

  def masking_actions(self, text, word_type=None, method=None):
    
    gender_word = text
    
    if word_type=="female_name":
      opp_gender_word = list(self.female_names)[self.random.randint(0, len(self.female_names))]
    
    elif word_type=="male_name":
      opp_gender_word = list(self.male_names)[self.random.randint(0, len(self.male_names))]

    elif word_type=="female_nonname":
      opp_gender_word = self.f2m[gender_word]
    
    elif word_type=="male_nonname":
      opp_gender_word = self.m2f[gender_word]
  
    if method=="naive_masking": #mask with config["naive_mask_token"]
      text = self.config["naive_mask_token"]

    elif method=="smart_fixed_masking":

      phrases = [
              "both [1] and [2]", 
              "[1] and [2]", 
              "[1] or [2]", 
              "either [1] or [2]"
      ]

      phrase = phrases[self.config["smart_fixed_mask_index"]]
      words = [gender_word, opp_gender_word]
      self.random.shuffle(words)
      phrase = phrase.replace("[1]", words[0])
      phrase = phrase.replace("[2]", words[1])
      text = phrase

    elif method=="smart_random_masking": #mask with smart phrases
      
      phrases = [
              "both [1] and [2]", 
              "[1] and [2]", 
              "[1] or [2]", 
              "either [1] or [2]"
      ]

      phrase = phrases[self.random.randint(0,len(phrases)-1)]
      words = [gender_word, opp_gender_word]
      self.random.shuffle(words)
      phrase = phrase.replace("[1]", words[0])
      phrase = phrase.replace("[2]", words[1])
      text = phrase

    elif method=="female_first_smart_random_masking": #mask with smart phrases with females first
      
      phrases = [
              "both [1] and [2]", 
              "[1] and [2]", 
              "[1] or [2]", 
              "either [1] or [2]"
      ]

      phrase = phrases[self.random.randint(0,len(phrases)-1)]
      
      words = [gender_word, opp_gender_word]
      self.random.shuffle(words)
      
      if word_type == "female_name" or word_type =="female_nonname":
        phrase = phrase.replace("[1]", gender_word)
        phrase = phrase.replace("[2]", opp_gender_word)
      else:
        phrase = phrase.replace("[1]", opp_gender_word)
        phrase = phrase.replace("[2]", gender_word)
        text = phrase
    
    elif method=="female_first_smart_fixed_masking": #mask with smart phrases with females first
      
      phrases = [
              "both [1] and [2]", 
              "[1] and [2]", 
              "[1] or [2]", 
              "either [1] or [2]"
      ]

      phrase = phrases[self.config["female_first_smart_fixed_mask_index"]]
      words = [gender_word, opp_gender_word]
      self.random.shuffle(words)
      
      if word_type == "female_name" or word_type =="female_nonname":
        phrase = phrase.replace("[1]", gender_word)
        phrase = phrase.replace("[2]", opp_gender_word)
      else:
        phrase = phrase.replace("[1]", opp_gender_word)
        phrase = phrase.replace("[2]", gender_word)
        text = phrase
    
    elif method=="neutral_masking":
      if(len(self.any2neutral[gender_word])!=0):
        text = self.any2neutral[gender_word][self.random.randint(0, len(self.any2neutral[gender_word])-1)]
    
    self.logger.debug("{} => {}".format(gender_word, text))
    self.swaps.append((gender_word, text))
    self.new_words.append(text)
    return text

  def process_batch(self, batch_data, fields_to_mask):
    for feild in fields_to_mask:
      for i in range(len(batch_data[feild])):
        batch_data[feild][i] = self.scan(text=batch_data[feild][i])
    return batch_data

  def modify_data(self, dataset):
    
    if "load_saved_data" in self.config and self.config["load_saved_data"]:
      data_output_path = os.path.join(self.root, "data_debias/data", "transformed_data_"+str(self.config["experiment_id"])+".h5")
      if(os.path.exists(data_output_path)):
        dataset = dataset.load_from_disk(data_output_path)
        words_output_path = os.path.join(self.root, "data_debias/data", "transformed_data_"+str(self.config["experiment_id"])+".pkl")
        with open(words_output_path, 'rb') as f:
          new_words = pickle.load(f)
        self.logger.warning("Reusing data from {}".format(data_output_path))
        return dataset, new_words
        
    args = {
              "fields_to_mask":["text"]
          }

    dataset = dataset.map(self.process_batch, fn_kwargs=args, batched=True)
    
    output_path = os.path.join(self.root, "data_debias", self.config["persistence_dir"], "transformed_data_stats_"+str(self.config["experiment_id"])+".json")  
    with open(output_path, "w") as f:
      d = {
        "words_affected":self.words_scanned,
        "rows_affected":self.rows_changed,
        "fraction_rows_affected":self.rows_changed/len(dataset)
      }
      f.write(json.dumps(d))

    self.logger.debug("Number of Words Neutralized: {} |  Number of Entries Neutralised: {} | Total Entries: {}".format(self.words_scanned, self.rows_changed, len(dataset)))
    self.wandb.log(d)

    output_path = os.path.join(self.root, "data_debias", self.config["persistence_dir"], "transformed_data_words_"+str(self.config["experiment_id"])+".csv")  
    with open(output_path, 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=',')
      writer.writerow(['before', 'after'])
      self.swaps = list(set(self.swaps))
      for swap in self.swaps:
        writer.writerow(swap)

    df = pd.DataFrame.from_records(list(dict(Counter(self.swaps)).items()), columns=['swap','count']).sort_values(by="count")
    self.wandb.log({"swaps":df})

    df = pd.DataFrame.from_records(list(dict(Counter([x[0] for x in self.swaps])).items()), columns=['swap','count']).sort_values(by="count")
    self.wandb.log({"words":df})

    df = pd.DataFrame.from_records(list(dict(Counter(self.new_words)).items()), columns=['word','count']).sort_values(by="count")
    self.wandb.log({"replacements":df})

    self.words_scanned=0
    self.rows_changed=0
    self.swaps = []
    
    new_words = self.new_words
    self.new_words = []
  
    if self.config["save_data"]:
      data_output_path = os.path.join(self.root, "data_debias/data", self.config["dataset"] + "_" + str(self.config["sample"]) + "_" + self.config["intervention"] + ".h5")
      dataset.save_to_disk(data_output_path)
      words_output_path = os.path.join(self.root, "data_debias/data", self.config["dataset"] + "_" + str(self.config["sample"]) + "_" + self.config["intervention"] + ".pkl")
      with open(words_output_path, 'wb') as f:
        pickle.dump(list(set(new_words)), f)
    
    return dataset, new_words

if __name__=="__main__":

  experiments = [
      {
        "name_mask_method":"smart_random_masking",
        "nonname_mask_method":"smart_random_masking",
        "naive_mask_token":"person",
        "seed":701,
        "persistence_dir":"src/logs",
        "save_data":True,
        "experiment_id":1,
        "load_saved_data":True
      }
    ]
    

  for config in experiments:
    print(config)
    dataTransformer = DataTransformer(config, "/Users/Himanshu/Developer/")
    dataset = load_dataset("csv", data_files="ssd.csv", split="train")
    dataset = dataset.rename_column("context", "text")
    dataset = dataset.shuffle(seed=701).select(range(10))
    dataset, new_words = dataTransformer.modify_data(dataset)
    print(new_words)
  
