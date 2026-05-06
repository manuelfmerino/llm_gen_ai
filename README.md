# llm_gen_ai
Repository for a RAG-assisted LLM for a question/answer chatbot that retrieves context from an internal policy file. It uses DPR libraries from HuggingFace (RAG_policy_file.ipynb).

The langchain_project.ipynb notebook includes code for a langchain-based project.

I'm using IBM's watsonx.ai lite support for the WatsonMachineLearning resource to run simple LLM tasks for free. This requires creating an account on IBM cloud and deploying a WatsonMachineLearning resource (see https://medium.com/the-power-of-ai/ibm-watsonx-ai-the-interface-and-api-e8e1c7227358). The user's authentication details (API key, URL for the service and project id) should be included in an .env file for obvious security reasons. The format should be:

WATSONX_URL=url_to_service.com
WATSONX_APIKEY=apikey
WATSONX_PROJECT_ID=projectid
