---
title: "Porting Jukebox's music generation model to transformers"
thumbnail: /blog/assets/07_porting_fsmt/thumbnail.png
---

<h1>Porting Jukebox's music generation model to transformers</h1>

<div class="blog-metadata">
    <small>Published November 3, 2020.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/porting-jukebox.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/stas">
        <img class="avatar avatar-user" src="/blog/assets/07_porting_fsmt/stas-bekman-300x300.jpg">
        <div class="bfc">
            <code>stas</code>
            <span class="fullname">Arthur Zucker</span>
            <span class="bg-gray-100 rounded px-1 text-gray-600 text-sm font-mono">guest</span>
        </div>
    </a>
</div>

# Introduction 

A few tutorials are available on the blog, but I had a hard time finding one that would treat the addition of a very complicated model. The [Sharing a cutom model](https://huggingface.co/docs/transformers/custom_models) tutorial is very helpful, but it only deals with pretty simple models that do not need pretraining. As we will be adding a complex model, we will need a deep understanding of how the library works. The most similar models added to the library are FLAVA and Wav2Vec2. Finding a similar model is already a complicated task if you don't know the library well. In that case you should ask @sylvain or @lysandre or anyone that has been in the team for a bit of time. I got my advise form @lucile and @patrick! 

An interesting and challenging part of adding a model is making sure that the training and the pretraining can work. This means that we have to take into account the tasks that are solved at training time. The VQ-VAE does not involve masking, but the transformer solve a nex-token prediction. I will not really dwelve into that for now since my initial task is just to prepare the model for inference. 


We will be creating the following classes: 

- JukeboxPreTrainedModel
- JukeboxForPreTraining
- JukeboxTokenizer          (will process all the text input)
- JukeboxFeatureExtractor   (or JukeboxCodebookFeatureExtractor as the feature extractor part will only be applied to the input of the codebook)
- JukeboxProcessor
- JukeboxConfig
- JukeboxOutput? 
- JukeboxAttention          (sclable transformers attention)
- JukeboxSelfAttention? 
- JukeboxTimeEmbeddings?    (Timing signal is included but HOW? )
- JukeboxEmbeddings         (construct the embeddings from the various tokens, should also be processing the timing conditioning? )
- JukeboxForPretrainingOutput
- JukeboxLosses             (this is very important as it will be used in the pretraining, during training the losses will be directly obtained from the transformer and the codebook will be frozen )
- JukeboxModelOutput


https://github.com/huggingface/transformers/blob/main/src/transformers/models/rag/modeling_rag.py is also a pretty complete guide and had similar issues? With the retreiver that is not *clean*. Wav2Vec2 also has a quantizer and is really similar to Jukebox in the way that the audio is quantized. 
# A. Setting up

## A.1 Create a fork and a branch
The firs step is to create a new fork of the hugging face transformers library. You will use this fork to develop your model and then ask for a pull request to be able to actually merge your changes to the official repository. You also need to create a branch, as it is common practice and avoids possible merge conficts when the main branch is updated. 

You need to make sure that your branch is always up to date with the official repository, thus you have to set the pull upstream to main using the following command : ...

## A.2 Clone the repository, setup

Clone your forked repository, switch to your branch, create a new conda/venv environment : `conda create -n jukebox python=3.8` for development and install the transformers library using `pip install -e ".[dev]"`.

## A.3 Clone the other repository
Here, I will be working on openAI's Jukebox model. Thus I will clone the official repository and install the dependencies. We will be porting that model to the transformers library. **As the model is huge, we will be working first on a dummy integration.** This means that we will create a very small version of the model to debug and test our new implementation. 

## A.4 Create template files 

Using the amazing `transformers-cli add-new-model` command every required file will be created based on a template. 

<details>
  <summary> Click to expand and see how I filled the form for Jukebox </summary>

    ```
    > transformers-cli add-new-model
    modelname [BrandNewBERT]: Jukebox
    uppercase_modelname [BRAND_NEW_BERT]: JUKEBOX
    lowercase_modelname [brand_new_bert]: jukebox
    camelcase_modelname [BrandNewBert]: Jukebox
    authors [The HuggingFace Team]: The HuggingFace Team
    checkpoint_identifier [brand-new-bert-base-cased]: huggingface/jukebox-base-cased
    Select tokenizer_type:
    1 - Based on BERT
    2 - Based on BART
    3 - Standalone
    Choose from 1, 2, 3 [1]: 3 
    Select generate_tensorflow_pytorch_and_flax:
    1 - PyTorch, TensorFlow and Flax
    2 - PyTorch & TensorFlow
    3 - PyTorch & Flax
    4 - TensorFlow & Flax
    5 - PyTorch
    6 - TensorFlow
    7 - Flax
    Choose from 1, 2, 3, 4, 5, 6, 7 [1]: 5
    Select is_encoder_decoder_model:
    1 - True
    2 - False
    Choose from 1, 2 [1]: 2
    ```
</details>

The following files would have been automatically created : 

* [`src/transformers/configuration_jukebox.py`](https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/src/transformers/configuration_jukebox.py) -  a short configuration class.
* [`src/transformers/convert_jukebox_original_pytorch_checkpoint_to_pytorch.py`](https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/src/transformers/convert_jukebox_original_pytorch_checkpoint_to_pytorch.py) - a complex conversion script. 
* [`src/transformers/modeling_jukebox.py`](https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/src/transformers/modeling_jukebox.py) - this is where the model architecture is implemented.
* [`src/transformers/tokenization_jukebox.py`](https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/src/transformers/tokenization_jukebox.py) - a tokenizer code.
* [`tests/test_modeling_jukebox.py`](https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/tests/test_modeling_jukebox.py) - model tests.
* [`tests/test_tokenization_jukebox.py`](https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/tests/test_tokenization_jukebox.py) - tokenizer tests.
* [`docs/source/model_doc/jukebox.rst`](https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/docs/source/model_doc/jukebox.rst) - a doc file.

An other command, `transformers-cli add-new-model-like` can also be used. This command is more up to date, and should you require a processor, it will be more convenient. It allows you to have all the template files that you will have to implement, and should help you designing the tests and better understand the philosophy of our library.


If you choose to add the model using `transformers-cli add-new-model-like`, as I did given that Jukebox is a decoder architecture, you would get the following : 



```
transformers-cli add-new-model-like
What is the model you would like to duplicate? gpt2
What is the name for your new model? Jukebox
What identifier would you like to use for the model type of this model?  [jukebox] jukebox
What name would you like to use for the module of this model?  [jukebox] 
What prefix (camel-cased) would you like to use for the model classes of this model?  [Jukebox] 
What prefix (upper-cased) would you like to use for the constants relative to this model?  [JUKEBOX] 
What will be the name of the config class for this model?  [JukeboxConfig] 
Please give a checkpoint identifier (on the model Hub) for this new model. huggingface/jukebox-dummy
Will your new model use the same processing class as gpt2 (GPT2Tokenizer)? No
What will be the name of the tokenizer class for this model?  [JukeboxTokenizer] 
Should we add # Copied from statements when creating the new modeling file?  [yes] no
Should we add a version of your new model in all the frameworks implemented by gpt2 (['pt', 'tf', 'flax'])?  [yes] no
Please enter the list of framworks you want (pt, tf, flax) separated by spaces pt
The tests for symbolic tracing with torch.fx were disabled, you can add those once symbolic tracing works for your new model.
The model you picked has the same name for the model type and the checkpoint name (gpt2). As a result, it's possible some places where the new checkpoint should be, you have jukebox instead. You should search for all instances of jukebox in the new files and check they're not badly used as checkpoints.
The constants at the start of the new tokenizer file created needs to be manually fixed. If your new model has a tokenizer fast, you will also need to manually add the converter in the `SLOW_TO_FAST_CONVERTERS` constant of `convert_slow_tokenizer.py`.
```

Now that you are all setup, we are going to start actualy coding. 
# Implementation 
In this section, I will describe my thinking process, and the various implementation choices that I made. 
## B.1 Creating the Tokenizers
As the inputs will first be processed by the tokenizer, it is a very important part of the implementation. You will also be able to easily test it against the original repository's code. 

Usually, if the model you are adding to the library is conventional, you probably won't even have to worry about implemnenting the tokenizer as you can re-use existing ones. Depending on the input type of your model, you can find a tokenizer in the Hugging Face Tokenizer library.

Here, the model we are adding is pretty complex and new, and it requires a particular design.
The JukeboxTokenizers needs to process the conditioning informations, which are the genres, the artists and the lyrics. 
Taking a closer look at the paper, it appears that 3 different dictionaries have to be used for each of the conditionings : 
- A dictionary which maps the possible artists to a unique ID 
- A dictionary which maps the possible genres to a unique ID
- A dictionary which maps accepted characeters to a unique ID for the lyrics. 

The philosophy of the transfomer library is 
> One model one file

Thus the three dictionaries have to be merged in a single new dictionary, which should have a key for the artists, the genres and the lyrics. Once I created this merged dictionnary, I uploaded it to a new hugging face repository at [ArthurZ/jukebox](https://huggingface.co/ArthurZ/jukebox). 

### OpenAI's tokenizer
The tokenizer also needs to take into account the duration of the audio that will be generated as this will influence the selection of the lyric tokens and the overall generated music. We also need it to generate outputs that are similat to the equivalent original toknizer in OpenAI's repository.
OpenAI used a `Labeller` which had a `ArtistAndGenreProcessor` and a `LyricProcessor`. I copy pasted most of the functions and subclasses related to the `Labeler` and created the `JukeboxTokenizer` class. 

### Porting the tokenizer 
When implementing the tokenizer, I implementing only the relevent function that were in the file created by the `transformer-cli`command, following the path of the inputs when the tokenizer is called. 
First,  the inputs have to be tokenized. The `tokenize` function will go through the next steps : 
1. `prepare_for_tokenization` : The first step is to preprocess the raw strings to remove unwanted characters and normalize the letters. The artist and genre inputs are normalized using the original `_normalize` function from the jukebox repo. The genres are then split, and the lyrics are normalized using  `BertNormalizer().normalize_str` from the toknizer library. 
2. `_tokenize`: The next step is to convert the string to tokens. The artists and genres should already be tokens on their own, while the lyrics need to be converted to a list of characters. This will return 3 list of tokens. 

Then, the tokens need to be converted to ids using the dictionary. The `__convert_token_to_id` function needs to be implemented. It will simply match the tokens with corresponding ids, and extract only the relevant lyrics tokens, which will focus on the most relevant tokens to the sequence. (For now I left it here, but as the relevent lyrics change during the auto regressive generation, it could be moved).

The implementation of the fast tokenizer also requires us to implement a `Converter` which would convert the slow tokenizer to a fast one. As it is not our priority, let's leave that for when our huge model will be training and focus on the next task : implementing the tests. 


## B.1.bis Creating the Tokenizer tests
This will allow us to make sure that the tokenizer gives coherent and similar results to the ones in the original repository. 

## B.2 Creating the configuration file
It is very important to be aware of the various hyperparameters available to the model. Moreover, implementing this first will allow us to test dummy and small models later on. 

## B.3 Creating the model 

## B.4 Creating tests

## B.5 Converting the weights and run full scale tests

## B.6 Creating the documentation 









Thank you for reading!
