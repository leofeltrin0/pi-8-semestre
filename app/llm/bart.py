from huggingface_hub import login, hf_hub_download

# Substitua 'YOUR_TOKEN' pelo token gerado
login(token='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')

# Baixar o modelo ou outros arquivos necessários
model_path = hf_hub_download(repo_id='meta-llama/Meta-Llama-3.1-8B-Instruct', filename='pytorch_model.bin')
config_path = hf_hub_download(repo_id='meta-llama/Meta-Llama-3.1-8B-Instruct', filename='config.json')
tokenizer_path = hf_hub_download(repo_id='meta-llama/Meta-Llama-3.1-8B-Instruct', filename='tokenizer.json')
