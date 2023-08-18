import together 
import os
from dotenv import load_dotenv, dotenv_values
import logging
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env
import json
import textwrap
from langchain import PromptTemplate,  LLMChain

#os.environ["TOGETHER_API_KEY"] = "b05aa6a73de99895c1f9c89cd0e0013d5ba22d7720a737bb7c106f3d417b8d48"

load_dotenv()

# set your API key
#together.api_key = os.environ["TOGETHER_API_KEY"]
together.api_key = os.getenv("TOGETHER_API_KEY")
print(together.api_key)


# list available models and descriptons
models = together.Models.list()

# print the first model's name
#print(models[3]['name']), print(models[51]['name'])

for idx, model in enumerate(models):
 print(idx, model['name'])

 

together.Models.start("togethercomputer/llama-2-7b-chat")




class TogetherLLM(LLM):
    """Together large language models."""

    model: str = "togethercomputer/llama-2-7b-chat"
    """model endpoint to use"""

    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        text = output['output']['choices'][0]['text']
        return text




B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")


def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        print(wrapped_text +'\n\n')
        # return assistant_text


llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature=0.1,
    max_tokens=512
)

system_prompt = "You are a capable large language model. You are going to do some named entity recognition and information extraction. Your task is to extract requested company information from a text and return the information in JSON format"
instruction = """
[1]Retrieve any companies or organizations names within the text. When there are more than one, return them in array format.
[2]For url, return any website link that is mentioned in the text. If there are no website links in the text, return nothing.
 [3]Detect any dates mentioned in the text, determine the day, month and the year. It could be date when this text is established or any date of an event mentioned in the text. Then return the date in Month-Day-Year with leading zeros format. Sometimes, you will need to convert months into a numeric value. Ignore days of the week. Dates doesn't have to be relevant to the company. If there are more than one date, return in an array.

[4]Return name of people mentioned in the text, do not return company name. Return their firstname and lastname. If there are no names mentioned in the text. Return null. When there are more than one person mentioned, return their names in an array.
[5]Return any places mentioned in the text, can be city, town, state, province, or country. Do not return address of the company if it is not in the text. If there is none in the text, return nothing.
[6]Determine what language is used to write the text, then return the kind of language used.
   Make sure your response is in JSON format with fields company, url, language, date, location, name. Do not include any information that is not in a JSON format.
    :\n\n {text}"""
template = get_prompt(instruction, system_prompt)
print(template)

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)


text = """
"""
output = llm_chain.run(text)

print(output)