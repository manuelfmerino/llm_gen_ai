from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import (
    WatsonxLLM,
)

import os


def llm_model(
    prompt_txt, model_id="mistralai/mistral-small-3-1-24b-instruct-2503", params=None
):
    """Function to initialize and wrap a model using IBM's watsonx.ai platform to be used with LangChain."""

    default_params = {
        "max_new_tokens": 256,
        "min_new_tokens": 0,
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 1,
    }

    if params:
        default_params.update(params)

    # Map parameters to IBM watsonx.ai API
    parameters = {
        GenParams.MAX_NEW_TOKENS: default_params["max_new_tokens"],
        GenParams.MIN_NEW_TOKENS: default_params["min_new_tokens"],
        GenParams.TEMPERATURE: default_params["temperature"],
        GenParams.TOP_P: default_params["top_p"],
        GenParams.TOP_K: default_params["top_k"],
    }

    # Define cloud credentials
    credentials = {
        "url": os.getenv("WATSONX_URL"),
        "apikey": os.getenv("WATSONX_APIKEY"),
    }
    project_id = os.getenv("WATSONX_PROJECT_ID")

    # Initialize model
    model = Model(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id,
    )

    # Wrap model into standard LLM interface for LangChain usage, converting it into a chat model
    llm_api = WatsonxLLM(model=model)

    return llm_api
