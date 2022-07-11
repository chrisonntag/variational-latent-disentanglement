# Structure of latent spaces in Variational Autoencoders
## Usage

The ```train.py``` script is the main starting point for training all implemented VAEs in this project. After 
training, each Encoder, Decoder and the model as a whole are saved to the ```experiments/``` folder in the 
projects root in the folling structure

```
experiments/
    RUN_NAME (e.g. conditionalVAE)/
        MODEL_NAME (e.g. 1333_live-similar-big-child)
            model/
                encoder/
                decoder/
                model/
            history.pckl
            params.pckl
            params.txt
```

The model names should be changed in the future in order to reflect some of the important hyperparameters used in this model in the directory name. 

The ```util``` package offers the Experiment class used for saving, which can also be used for loading the Tensorflow model and the used parameters 
by passing the model and run name.

```
experiment = Experiment(name=MODEL_NAME, base_path="experiments/conditionalVAE")        
base_model = experiment.load_model()
params = {'input_dim': (28, 28, 1), 'z_dim': experiment.params['latent_dim'], 'label_dim': 10, 'beta': experiment.params['beta']}            
model = ConditionalVAE.from_saved_model(base_model, params)
```

Depending on the used architecture, one may call the ```from_saved_model()``` method from differet classes (TODO: Implement a parental class, where all implementations can inherit this method). 
It may be easier to load all experiments from a run (different hyperparameters for the same architecture) at once into a dictionary. 
In the following example we use the ```load_experiments()``` function from the ```experiment``` module to load several experiments with certain parameters at once: 

```
run_name = 'branchedClassifier'
hyperparameters = {'latent_dim': 4}
experiments = load_experiments(base_path="experiments/"+run_name, with_params=hyperparameters)
 
```

These can then be used as shown above.

## Why

Reconstructing and understanding decisions made by machine learning models becomes more and more
important since used in areas that have great impact on societies and may directly affect people’s lives.

This project’s goal is to contribute to the understanding and visualization of latent spaces in Variational
Autoencoders based on labour market data. 

The main idea of using Autoencoders is in general that by
reducing dimensions from the input space while trying to resemble the input from this lower dimensional
representation at the same time leads to a compressed version of usually high dimensional input spaces that
contain just as much information to describe the input properly and can thus be further investigated and
used for various applications. 
However, due to the loss of any intrinsical meaning of latent dimensions,
interpretation is not intuitively possible – even in human-understandable two-dimensional spaces. In
addition to that, variables do not correspond with individual features of the input space directly but
may resemble combinations of multiple ones. 
Therefore, imposing structure on latent spaces may help
to further understand hidden relationships in data. This project will focus on two individual parts:
The review, proposal and qualitative evaluation of methods for disentanglement and the building of a
hierarchical structure on latent spaces and possible visualizations that help users to understand abstract
representations of data which should also be interpretable by non-expert users.


