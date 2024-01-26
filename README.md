# Structure of latent spaces in Variational Autoencoders
Reconstructing and understanding decisions made by machine learning models becomes more and more
important since used in areas that have great impact on societies and may directly affect people’s lives.

This project’s goal is to contribute to the understanding and visualization of latent spaces in Variational
Autoencoders in order to advance knowledge on trustworthy AI topics.

The main idea of using Autoencoders in general is that by
reducing dimensions from the input space while trying to resemble the input from this lower dimensional
representation at the same time leads to a compressed version of usually high dimensional input spaces that
contain just as much information to describe the input properly and can thus be further investigated and
used for various applications. 

However, due to the loss of any intrinsical meaning of latent dimensions,
interpretation is not intuitively possible – even in human-understandable two-dimensional spaces. In
addition to that, variables do not correspond with individual features of the input space directly but
may resemble combinations of multiple ones. 

Therefore, imposing structure on latent spaces may help 
to further understand hidden relationships in data. This project focuses on two individual parts: 

- Imposing structure to the latent space by adapting the training process and architecture of the model
- Creating an interactive visualization for exploring the latent space

## Architectures
### Conditional VAE
In a conditional VAE, the input sample is concatenated with the class label during the encoding-decoding process. 
This means the encoder samples from ```P(z|x, c)``` instead of ```P(z|x)```. 

![Encoder of a Conditional VAE](https://github.com/chrisonntag/variational-latent-disentanglement/blob/main/assets/images/architecture/conditional_encoder.png?raw=true)

### Branched Classifier
In this architecture, the idea was to introduce a third loss, that optimizes a separate branch as an classifier.

![Encoder of a VAE with a branched classifier](https://github.com/chrisonntag/variational-latent-disentanglement/blob/main/assets/images/architecture/model_encoder.png?raw=true)

## Interactive Visualization
The interactive visualization is built with DASH, which allows interactive elements in Jupyter Notebooks. 
User can traverse through the latent space of trained models with sliders, while observing changes in the SPLOM upon changes in dimension or training hyperparameters. 

![Interactive visualization of VAEs](https://github.com/chrisonntag/variational-latent-disentanglement/blob/main/assets/images/vis.png?raw=true)

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

## Disclaimer
This is a WIP research project and no released paper. This repo is meant for exploration purposes. 


