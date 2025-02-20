import os
import oci
import asyncio
from concurrent.futures import ThreadPoolExecutor

from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
)


class OCICohereCommandRLLM:
    template_type = "cohere-command-r"

    def __init__(self, model_name="cohere.command-r-plus"):
        """
        Initialize the object with the model attribute set to "cohere.command-r-plus".
        """
        self.model = model_name
        self.get_client()

    def get_client(self):
        """
        Generates the Oracle Cloud Infrastructure (OCI) client based on the authentication type.
        Initializes the client with the appropriate configuration, signer, service endpoint, retry strategy,
        and timeout based on the given authentication type.
        """
        endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        config = {"region": REGION}
        signer = None
        if AUTH_TYPE == "API_KEY":
            config = oci.config.from_file(profile_name=OCI_PROFILE)
            config["region"] = REGION
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=config,
                service_endpoint=endpoint,
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240),
            )
        elif AUTH_TYPE == "INSTANCE_PRINCIPAL":
            signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=config,
                signer=signer,
                service_endpoint=endpoint,
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240),
            )
        elif AUTH_TYPE == "RESOURCE_PRINCIPAL":
            signer = oci.auth.signers.get_resource_principals_signer()
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=config,
                signer=signer,
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240),
            )
        else:
            # log.error(
            #     "Please provide a valid OCI_AUTH_TYPE from the following : API_KEY, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL")
            print(
                "Please provide a valid OCI_AUTH_TYPE from the following : API_KEY, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL"
            )

    def generate_answer(self, preamble, prompt, documents):
        """
        Generate the chat response using the provided preamble, prompt, and documents.

        Parameters:
            preamble (str): The text to set as the preamble override.
            prompt (str): The text prompt for the chat response.
            documents (list): A list of documents to consider during chat generation.

        Returns:
            str: The generated chat response text.
        """
        # profile = OCI_PROFILE
        compartment_id = COMPARTMENT_ID
        generative_ai_inference_client = self.client
        chat_detail = oci.generative_ai_inference.models.ChatDetails()
        chat_request = oci.generative_ai_inference.models.CohereChatRequest(
            preamble_override=preamble
        )
        chat_request.message = prompt

        chat_request.max_tokens = 4000
        chat_request.is_stream = False
        chat_request.temperature = 0.00
        chat_request.top_p = 0.7
        chat_request.top_k = 1  # Only support topK within [0, 500]
        chat_request.frequency_penalty = 1.0
        # chat_request.prompt_truncation = 'AUTO_PRESERVE_ORDER'

        chat_request.documents = documents

        chat_detail.serving_mode = (oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.model))

        chat_detail.compartment_id = compartment_id
        chat_detail.chat_request = chat_request

        chat_response = generative_ai_inference_client.chat(chat_detail)

        chat_response_vars = vars(chat_response)
        resp_json = json.loads(str(chat_response_vars["data"]))
        res = resp_json["chat_response"]["text"]
        # log.debug(res)
        return res

async def llm_model_ociCohereRLLM(prompt, preamble, documents):
    commandr = OCICohereCommandRLLM()
    return await commandr.generate_answer(preamble=preamble, prompt=prompt, documents=documents)

# multithread llm call with blocking functions
async def llm_model_ociCohereRLLM_multithread(prompt, preamble, documents):
    commandr = OCICohereCommandRLLM()

    # Define a blocking function to run in a thread
    def generate_answer():
        return commandr.generate_answer(preamble=preamble, prompt=prompt, documents=documents)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, generate_answer)

    return result

