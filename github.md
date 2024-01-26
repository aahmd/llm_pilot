# Resources:

## Machine Learning (ML) and LLM's (Large Language Models):

### Basics:

### Fine-tuning:

#### RLHF (Reinforcement Learning from Human Feedback):

- https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback
- https://arxiv.org/abs/2203.02155 (Training language models to follow instructions with human feedback)
- https://arxiv.org/abs/2310.06452 (Understanding the Effects of RLHF on LLM Generalisation and Diversity)

#### DPO (Direct Preference Optimization):

 a 'new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning.'

- https://arxiv.org/html/2305.18290v2 (Direct Preference Optimization: Your Language Model is Secretly a Reward Model)
- https://towardsdatascience.com/fine-tune-a-mistral-7b-model-with-direct-preference-optimization-708042745aac


#### RAG (Retrieval-Augmented Generation):

## Langchain:

### Vectorstores:
Store and search your data:
- https://python.langchain.com/docs/modules/data_connection/vectorstores/ (conceptual overview)
- https://integrations.langchain.com/vectorstores (available options)

### Loaders:

- used to load data from some source
- helpful for Retrieval-Augmented Generation (RAG)


```python
from langchain_community.document_loaders import WebBaseLoader

# pages that are hard to parse, require login will require more handling than this simple example:
loader = WebBaseLoader(
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9974218/pdf/12553_2023_Article_738.pdf"
)
data = loader.load()
```
- https://js.langchain.com/docs/modules/data_connection/document_loaders/ (conceptual overview)
- https://python.langchain.com/docs/integrations/document_loaders/ (available options - there are many!)

### Prompt and PromptTemplates:
Tools to help with your prompt:
- https://python.langchain.com/docs/modules/model_io/prompts/ (conceptual overview)

## Microsoft/AOAI:

### Azure KeyVault:

- enables safe handling of secrets (api keys, access tokens, etc.)
- helps to obfuscate these values in your code
- https://learn.microsoft.com/en-us/azure/key-vault/general/overview (conceptual overview)

### Azure Chat Playground:

- test basic functionality
- the Azure ML workspace may be easier to use after a point
- https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models (current models available)

### Azure ML:

#### Create a new environment and install some libraries (python):
- change the 'ENV' (all occurrences) below to some name you want to give the environment
- change the python version to 3.11 (or another) if needed
- this can be ran in a terminal, next to a notebook or within the notebook itself using `!` as a prefix
- a refresh of the page or kernel may be help resolve issues

```sh
# create a new environment:
conda create -n ENV python=3.10
# (when you run this command it will ask you 'y or n' to install; press enter to continue or enter y + enter)

# activate environment:
conda activate ENV

# install various python libraries we want to use:
pip install ipykernel openai langchain bs4 pandas tiktoken chromadb

# add kernel to UI:
python -m ipykernel install --user --name ENV --display-name "ENV"
```

#### Getting reproducible results:

What if you want to get the same result from a model with a particular prompt?

- https://cookbook.openai.com/examples/reproducible_outputs_with_the_seed_parameter
- https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/reproducible-output?tabs=pyton

#### Promptflow:

- https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/overview-what-is-prompt-flow?view=azureml-api-2

## Open Source Models:

### Huggingface:

- 'a platform where the machine learning community collaborates on models, datasets, and applications'
- https://huggingface.co/



## Other:

