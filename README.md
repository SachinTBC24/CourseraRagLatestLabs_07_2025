There are 2 kind of search technique available
1) keyword search
2) semantic search

   1) keyword search uses direct test search like traditional database and etc.
   2) semantic search did the vector search and find the nearest match
      It uses 2 technique
      a) ucledian distance
      b) cosine similarity

TF-IDF and BM25 those are 2 algorithm used for vectorization.
Eg: While TF-IDF heavily penalizes the cookbook chapters for having lower keyword density (0.5% vs. 5%), BM25's document length normalization reduces this penalty. Additionally, its term frequency saturation means those 30 mentions in the cookbook chapter still count significantly.

<img width="975" height="542" alt="image" src="https://github.com/user-attachments/assets/bebfc532-a545-43ec-87b9-60ac756e4807" />
<img width="1048" height="481" alt="Screenshot 2025-07-21 143059" src="https://github.com/user-attachments/assets/d2d981bf-fad8-40e5-afee-47d3e9654249" />
<img width="1032" height="474" alt="image" src="https://github.com/user-attachments/assets/8a21abdb-0994-4e54-9be7-25005be8245d" />
<img width="996" height="333" alt="image" src="https://github.com/user-attachments/assets/f603e6ef-e5fe-4fb8-8844-73d7fd930450" />

<img width="1391" height="888" alt="image" src="https://github.com/user-attachments/assets/d5152181-cf32-46c6-8750-3548c14eff5a" />

C1M3_Ungraded_Lab_2.ipynb go through this file for chunking strategy.

C1M3_Assignment.ipynb go through this file for Metadata filtering,  Semantic search, BM25 Serach, Hybrid search, Reranking with Weaviate API.

Another way that will be largely used in this modules is to pass a keyword dictionary as parameters. You need to pass it as **kwargs

kwargs = {"prompt": "Write a poem about a flying rabbit.", 'top_p': 0.7, 'temperature': 1.4, 'max_tokens': 100}
generate_with_single_input(**kwargs)

prompt: Input text for the model.
-> temperature: Controls randomness; lower values = more deterministic.
-> top_p: Controls diversity; higher values = more varied outputs.
-> max_new_tokens: Sets the maximum number of tokens in the response.

def generate_params_dict(
    prompt: str, 
    temperature: float = None, 
    role = 'user',
    top_p: float = None,
    max_tokens: int = 500,
    model: str = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
):
    """
    Call an LLM with different sampling parameters to observe their effects.
    
    Args:
        prompt: The text prompt to send to the model
        temperature: Controls randomness (lower = more deterministic)
        top_p: Controls diversity via nucleus sampling
        max_tokens: Maximum number of tokens to generate
        model: The model to use
        
    Returns:
        The LLM response
    """
    
    # Create the dictionary with the necessary parameters
    kwargs = {"prompt": prompt, 'role':role, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, 'model': model} 


    return kwargs

def generate_with_single_input(prompt: str, 
                               role: str = 'user', 
                               top_p: float = None, 
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str ="meta-llama/Llama-3.2-3B-Instruct-Turbo",
                               together_api_key = None,
                              **kwargs):
    
    if top_p is None:
        top_p = 'none'
    if temperature is None:
        temperature = 'none'

    payload = {
            "model": model,
            "messages": [{'role': role, 'content': prompt}],
            "top_p": top_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
                  }
    if (not together_api_key) and ('TOGETHER_API_KEY' not in os.environ):
        url = os.path.join('https://proxy.dlai.link/coursera_proxy/together', 'v1/chat/completions')   
        response = requests.post(url, json = payload, verify=False)
        if not response.ok:
            raise Exception(f"Error while calling LLM: f{response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}")
    else:
        if together_api_key is None:
            together_api_key = os.environ['TOGETHER_API_KEY']
        client = Together(api_key =  together_api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()
        json_dict['choices'][-1]['message']['role'] = json_dict['choices'][-1]['message']['role'].name.lower()
    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'], 'content': json_dict['choices'][-1]['message']['content']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict    
<img width="1819" height="897" alt="image" src="https://github.com/user-attachments/assets/e7732e78-a85f-4aa9-92ed-72d809babf1e" />





