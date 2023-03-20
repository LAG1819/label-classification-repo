# Tutorial 04: Classify Data with a pretrained BERT

In this tutorial you will find a short introduction on how to train and apply a pretrained BERT-model including Hyperparameteroptimization. 

To learn more about PyTorch for model customizations check out [PyTorch.org ](https://pytorch.org/).  
To learn more about the unlimited possibilities of pre-trained Machine Learning models please check out the HuggingFace community on [:hugs: HuggingFace.co](https://huggingface.co/)

# Table of Contents
1. [Start a complete Classification training](#start-classification-training)
2. [Customize number of classes](#customize-number-of-classes)
3. [Customize parameter of ressources](#customizer-parameter-of-ressources-usage)
4. [Add custom hyperparameter optimization techniques](#add-or-change-hyperparameter-optimization-techniques) 

# Start Classification training
There are two possible ways to start a classification training. One is simply by executing the main menu. Another option is to create a classification run that also allows customizations.
## Execute the main menu
* Start [main](https://github.com/LGHDM/ml-classification-repo/blob/main/__main__.py) in terminal and the main menu will show up.
  ```console
   python3 ml-classification-repo
  ```
* Select language: 
   ```Python3
      Please select Language.
      1: German 
      2: English
   1
   ```
* Execute data k-Means generation.
   ```Python3
      Please select an Option:
      (0) End session.
      (1) Execute full process.
      (2) Execute data crawling.
      (3) Execute data cleaning.
      (4) Execute topic extraction.
      (5) Execute automated labeling: k-Means. 
      (6) Execute automated labeling: Train and Apply Label Modell.
      (7) Execute classification: Train and Apply Classification Modell.
   7
   ```
* Select custom database - supported types are .feather, .xlsx or .csv.:
   ```Python3
      Take custom data? (y/n)
    n
   ```

## Create Classification run 
```Python3
from src.classifier.classifier_model import run as classifier_run

language = 'de'
textcolumn_to_learn_classification = 'TOPIC'

classifier_run(lang ='de', col = 'TOPIC')
```

# Customize number of classes
To change the number of classes the classifier needs to learn with one can simply create a classification run and set the parameter nbr_class

```Python3
from src.classifier.classifier_model import run as classifier_run

language = 'de'
textcolumn_to_learn_classification = 'TOPIC'
desired_number_classes = 3

classifier_run(lang ='de', col = 'TOPIC', nbr_class = desired_number_classes)
```

# Customizer parameter of ressources usage
There are several parameters to adapt for training the classification modell so that the computing capacities are used optimally and no errors occur (e.g. CUDA out of memory).  

Therefore the number of cpu and gpu to use for the training can be selected, as well as the number of workers for parallel data allocation by the DataLoader.
To change the number of hyperparameteroptimization trials and the iteration per optimization trial one can change the parameters num_samples_per_tune and num_training_iter.  

*Note: Please check your computational resources before setting these parameters!*
  
  
```Python3
from src.classifier.classifier_model import run as classifier_run

language = 'de'
textcolumn_to_learn_classification = 'TOPIC'
number_cpu = 4
number_gpu = 1
number_workers = 4
number_trials = 5
number_trial_iterations = 3

classifier_run(lang ='de', col = 'TOPIC', num_cpu = number_cpu, num_gpu = number_gpu, num_workers = number_workers, num_samples_per_tune = number_trials, num_training_iter = number_trial_iterations)
```

# Add or change hyperparameter optimization techniques
To change of add other hyperparameter optimization techniques please read the latest documentation and guidlines of [Ray.Tune](https://docs.ray.io/en/latest/tune/index.html) and generate customized optimization techniques according to them.   
To add a hyperparameter optimization technique it can simply added as parameter in the run function. 
```Python3
import ray
from ray import tune 
from ray.air.checkpoint import Checkpoint
from ray.air import session
from ray import air
from ray.tune import CLIReporter
from ray.tune.experiment.trial import Trial
from src.classifier.classifier_model import run as classifier_run

def my_hyperparameter_optimization_technique(lang:str, text_col:str, path_to_save_experiment:str,data_path:str, num_workers:int,num_samples = 1, num_cpu=2, num_gpu=1,  num_training_iter = 5,nbr_class=7):
    
    configuration_space = {
            "lr":tune.loguniform(1e-4,1e-1),
            "batch_size":tune.choice([2,4,6,8,12]),
            "epoch":tune.choice([3,5,7,10]),
            "lang":lang,
            "text_col":col,
            'n_class':nbr_class,
            'data_dir' : data_path,
            "num_workers": num_workers
        }
            
    random_search = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(_train_model),
            resources={"cpu": num_cpu, "gpu":num_gpu}
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            num_samples = num_samples,
        ),
        param_space=config_rand,
        run_config=air.RunConfig(local_dir=path_to_save_experiment, name="random_search_"+lang, stop={"training_iteration":  num_training_iter})
    )
    result = random_search.fit()
    best_result = result.get_best_result("accuracy", "max")
    return best_result

my_hpos = [("MyCustomHPO", my_hyperparameter_optimization_technique)]
classifier_run(lang ='de', col = 'TOPIC', list_of_hpo = my_hpos)
    
```