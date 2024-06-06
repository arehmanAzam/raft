import boto3
import json
from logconf import log_setup
import logging
from typing import Literal, Any
from langchain_community.llms import Bedrock
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain import LLMChain
from langchain_aws import ChatBedrock
from botocore.exceptions import ClientError
import ast
log_setup()
logger = logging.getLogger("bedrock_call")
llm=None
bedrock_client=None


def bedrock_inference(client,prompt : str):
    global bedrock_client,llm
    prompts=[]
    prompts.append(("system","You are a helpful question answerer who can provide an answer given a question and relevant context."))
    prompts.append(("user", "{prompt}"))
    chat_template = ChatPromptTemplate.from_messages(prompts)

    if llm == None and bedrock_client == None:
        bedrock_client = client
        llm = ChatBedrock(
            model_id='meta.llama3-70b-instruct-v1:0',
            client=client,
            model_kwargs={'temperature': 0.5}
        )

    chain = LLMChain(llm=llm,
                     prompt=chat_template,
                    )
    answer = chain.invoke({"prompt":prompt})
    logger.info(answer)
    return answer
def json_to_list(data: dict) -> list:
    """
    Helper function for helping format response questions to list
    """
    question_text = data['text']
    start_index = question_text.find('[') + 1  # Find the start of the list
    end_index = question_text.find(']')  # Find the end of the list
    questions_list_text = "["+question_text[start_index:end_index]+"]"
    questions_list_text = questions_list_text.strip()
    logger.info(questions_list_text)
    questions_list = json.loads(questions_list_text)
    return questions_list
def get_bedrock_client(region,aws_access_key_id,aws_secret_access_key):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region,aws_access_key_id = aws_access_key_id,
    aws_secret_access_key = aws_secret_access_key)
    return bedrock_client
def bed_generate_instructions_gen(client:Any, chunk: Any, x: int = 5, model: str = None) -> list[str]:
    """
    Generates `x` questions / use cases for `chunk`. Used when the input document is of general types
    `pdf`, `json`, or `txt`.
    """

    global bedrock_client,llm
    try:
        if llm==None and bedrock_client==None:
            bedrock_client=client
            llm = ChatBedrock(
                model_id='meta.llama3-70b-instruct-v1:0',
                client=client,
                model_kwargs={'temperature': 0.5}
            )

        chat_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a synthetic question-answer pair generator. Given a chunk of context about some topic(s), generate %s example questions a user could ask and would be answered using information from the chunk. For example, if the given context was a Wikipedia paragraph about the United States, an example question could be 'How many states are in the United States?'" % (x)),
                ("system", "The questions should be able to be answered in a few words or less. Include only the questions in your response."),
                ("user", str(chunk)),
                ("human","{input}")
            ]
        )
        chain = LLMChain(llm=llm,
                         prompt=chat_template,
                        )
        completion = chain.invoke({"input":"Give Questions in a form of python list"})
        questions=json_to_list(completion)
        logger.info(questions)
        return questions
    except Exception as e:
        logger.error(e)
        return None