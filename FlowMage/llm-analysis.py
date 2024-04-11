import argparse
import itertools
import json
import os
import pathlib
import time
from typing import Any, List, Optional, Union

from langchain.callbacks import get_openai_callback
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_core.language_models.base import BaseLanguageModel
from langchain.chat_models.base import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

#Fill this parameter with your Open AI API key to use ChatGPT 3.5 or ChatGPT 4.
os.environ["OPENAI_API_KEY"] = ""
#Fill this parameter with your application_credentials.json file to enable Gemini!
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

#Modify these parameters to appropriate values from your vertex-ai dashboard!  
PROJECT_ID = "firm-region-415309"  # @param {type:"string"}
REGION = "us-east4"

stateful_modules = [
    "acl", "lb", "nat64", "pnat", "IPortScanDetector", "IPolicer", "FlowRateLimiter", "ISourceCounter", "FlowHyperScan",
    "FlowIPLoadBalancer", "FlowCounter",
]


def _build_llama2_prompt(messages):
    start_prompt = "<s>[INST] "
    end_prompt = " [/INST]"

    conversation = []
    for index, message in enumerate(messages):
        parts = message.split(": ", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid message format: {message}")

        role, content = parts
        content = content.strip()

        if role.lower() == 'user':
            conversation.append(content)
        elif role.lower() == 'ai':
            conversation.append(f"{content}")
        elif role.lower() == 'function':
            raise ValueError('Llama 2 does not support function calls.')
        elif role.lower() == 'system':
            conversation.append(f"<<SYS>>\n{content}\n<</SYS>>\n\n")
        else:
            raise ValueError(f"Invalid message role: {role}")

    return start_prompt + ''.join(conversation) + end_prompt


model_configurations = {
    'gpt-3.5-turbo': {
        'model_name': 'gpt-3.5-turbo-1106',
        'type': 'openai',
        'args': {
            'response_format': {'type': 'json_object'},
            'seed': 5000,
        }
    },
    'gpt-4-turbo': {
        'model_name': 'gpt-4-1106-preview',
        'type': 'openai',
        'args': {
            'response_format': {'type': 'json_object'},
            'seed': 5000,
        }
    },
    'codellama-7b-instruct': {
        'model_name': 'codellama/CodeLlama-7b-Instruct-hf',
        'prompt_builder': _build_llama2_prompt,
        'max_length': 4096,
        'type': 'HF',
        'use_quantization': False
    },
    'codellama-13b-instruct': {
        'model_name': 'codellama/CodeLlama-13b-Instruct-hf',
        'prompt_builder': _build_llama2_prompt,
        'max_length': 4096,
        'type': 'HF',
        'use_quantization': False
    },
    'codellama-34b-instruct': {
        'model_name': 'codellama/CodeLlama-34b-Instruct-hf',
        'prompt_builder': _build_llama2_prompt,
        'use_quantization': True,
        'max_length': 65536,
        'type': 'HF'
    },
    'gemini-1-pro': {
        'model_name': 'gemini-pro',
        'type': 'google'
    }
}


def build_llm(model: str) -> BaseChatModel:
    if model_configurations[model]['type'] == 'HF':
        from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
        from langchain.chat_models.base import BaseChatModel
        from langchain.schema import BaseMessage, AIMessage, ChatResult, ChatGeneration, HumanMessage, SystemMessage
        from torch import cuda, bfloat16
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            pipeline,
            BitsAndBytesConfig
        )

        class ChatHF(BaseChatModel):
            model_name: str
            max_length: int
            use_quantization: bool
            temperature: float
            text_pipeline: Any
            prompt_func: Any

            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self.model_name = kwargs.get('model_name', "")
                self.max_length = kwargs.get('max_length', 4096)
                self.use_quantization = kwargs.get('use_quantization', False)
                self.temperature = kwargs.get('temperature', 0.001)
                self.prompt_func = kwargs.get('prompt_func', None)
                self.text_pipeline = self._initialize_text_pipeline()

            @property
            def _llm_type(self) -> str:
                return self.model_name

            def _initialize_text_pipeline(self):
                device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

                additional_model_kwargs = {}
                if self.use_quantization:
                    additional_model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=bfloat16
                    )

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    device_map='auto',
                    **additional_model_kwargs
                )
                # model.to(device)
                model.eval()

                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )

                text_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    trust_remote_code=True,
                    device_map='auto',
                    do_sample=True,
                    top_p=1,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    repetition_penalty=1,
                )

                return text_pipeline

            def _generate(
                    self,
                    messages: List[BaseMessage],
                    stop: Optional[List[str]] = None,
                    **kwargs: Any,
            ) -> ChatResult:
                chat_history = self._parse_chat_history(messages)
                input_text = self.prompt_func(chat_history)
                generation = self.text_pipeline(input_text)
                response_text = generation[0]['generated_text'][len(input_text):]
                message = AIMessage(content=response_text)
                return ChatResult(generations=[ChatGeneration(message=message)])

            async def _agenerate(
                    self,
                    messages: List[BaseMessage],
                    stop: Optional[List[str]] = None,
                    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
                    **kwargs: Any,
            ) -> ChatResult:
                raise NotImplementedError(
                    """HF doesn't support async requests at the moment."""
                )

            def _parse_message(self, role: str, text: str) -> str:
                return f"{role}: {text}"

            def _parse_chat_history(self, history: List[BaseMessage]) -> List[str]:
                """Parse a sequence of messages into history.

                Returns:
                    A tuple of a list of parsed messages and an instruction message for the model.
                """
                chat_history = []
                for message in history:
                    if isinstance(message, HumanMessage):
                        chat_history.append(self._parse_message("User", message.content))
                    if isinstance(message, AIMessage):
                        chat_history.append(self._parse_message("AI", message.content))
                    if isinstance(message, SystemMessage):
                        chat_history.append(self._parse_message("System", message.content))

                return chat_history

        return ChatHF(
            model_name=model_configurations[model]['model_name'],
            max_length=model_configurations[model]['max_length'],
            use_quantization=model_configurations[model]['use_quantization'],
            prompt_func=model_configurations[model]['prompt_builder'],
            temperature=0.001,
        )
    elif model_configurations[model]['type'] == 'openai':
        from langchain_openai import ChatOpenAI
        from langchain_community.callbacks import get_openai_callback

        return ChatOpenAI(
            model=model_configurations[model]['model_name'],
            model_kwargs=model_configurations[model]['args'],
            temperature=0.001,
        )
    elif model_configurations[model]['type'] == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
        from langchain_google_vertexai import ChatVertexAI
        from langchain_google_vertexai import VertexAI
        import vertexai

        vertexai.init(project=PROJECT_ID, location=REGION)
        return ChatVertexAI(
            model_name=model_configurations[model]['model_name'],
            convert_system_message_to_human=True,
            temperature=0.001
        )
    else:
        raise Exception(f"Type `{model_configurations[model]['type']}` for model `{model}` not supported!")


def find_merged_files(directory: str) -> dict:
    merged_files = {}

    for file in (p.resolve() for p in pathlib.Path(directory).glob(f"**/*-merged.txt")):
        merged_files[file.parent.name] = str(file)

    return merged_files


def extract_statefulness(model: str, name: str, src_file: str, it: int, results: dict) -> None:
    print(f"[IT #{it}] Extracting statefulness data for {name}...")

    messages = [
        ("system", "Your task is to analyze C or C++ code of a network function provided by the user. "
                   "The user will provide the header and the actual code of the network function. "
                   "You also act like a REST server, you only talk JSON, no natural language. "
                   "The user will provide the output format of the JSON."),
        ("system", "For each network function, the user will ask if it is stateful, "
                   "which means fetching and storing data per flow separately into a dedicated data structure. "
                   "A network function is stateful if a flow is identified by the 5-tuple, or a subset of them: "
                   "Source IP, Destination IP, Source Port, Destination Port, Protocol. "
                   "If the NF stores other types of data, it is not stateful."),
        ("human", "Here is the C or C++ code of the network function to analyze:\n{code}"),
        ("human", "Provide a JSON containing a single key 'statefulness' that describes if the NF is stateful or not "
                  "without any additional explanation. Possible values are: 'stateful', 'stateless'."),
    ]
    if model_configurations[model]['type'] == 'google':
        # From https://python.langchain.com/docs/integrations/chat/google_generative_ai
        # For regular chat conversations, messages must follow the human/ai/human/ai alternating pattern.
        # You may not provide 2 AI or human messages in sequence
        second_msg = messages[1]
        messages.pop(1)
        messages[0] = (messages[0][0], messages[0][1] + "\n" + second_msg[1])
        last_msg = messages[-1]
        messages.pop(-1)
        messages[1] = (messages[1][0], messages[1][1] + "\n" + last_msg[1])

    llm = build_llm(model)
    prompt_template = ChatPromptTemplate.from_messages(messages)
    llm_chain = (prompt_template | llm | JsonOutputParser())

    results[name][it]['statefulness'] = {
        'model_error': None,
        'result': None
    }
    with open(src_file, 'r') as file:
        try:
            with get_openai_callback() as cb:
                result = llm_chain.invoke({'code': file.read()}, config={'callbacks': [ConsoleCallbackHandler()]})
                results[name][it]['statefulness']['prompt_tokens'] = cb.prompt_tokens
                results[name][it]['statefulness']['completion_tokens'] = cb.completion_tokens
                results[name][it]['statefulness']['total_cost'] = cb.total_cost
                results[name][it]['statefulness']['result'] = result['statefulness']
        except Exception as e:
            results[name][it]['statefulness']['model_error'] = str(e)


def extract_read_write_intensity(model: str, name: str, src_file: str, it: int, results: dict) -> None:
    print(f"[IT #{it}] Extracting intensity data for {name}...")

    messages = [
        ("system", "Your task is to analyze C or C++ code of a stateful network function provided by the user. "
                   "The user will provide the header and the actual code of the network function. "
                   "You also act like a REST server, you only talk JSON, no natural language. "
                   "The user will provide the output format of the JSON."),
        ("system", "For each network function, the user will ask how often the states are being updated, "
                   "more specifically if states are being updated for all packets of a flow or "
                   "just once at the start of the flow. "
                   "A flow is identified by the 5-tuple, or a subset of them: "
                   "Source IP, Destination IP, Source Port, Destination Port, Protocol. "
                   "If the NF stores other types of data, they are not considered as a flow identifier."),
        ("human", "Here is the C or C++ code of the network function to analyze:\n{code}"),
        ("human", "Provide a JSON containing a single key 'intensity' that describes the intensity of the NF "
                  "without any additional explanation. Possible values are: "
                  "'per-packet' when states are being updated per packet of a flow, and "
                  "'per-flow' when states are being updated occasionally for each flow."),
    ]
    if model_configurations[model]['type'] == 'google':
        # From https://python.langchain.com/docs/integrations/chat/google_generative_ai
        # For regular chat conversations, messages must follow the human/ai/human/ai alternating pattern.
        # You may not provide 2 AI or human messages in sequence
        second_msg = messages[1]
        messages.pop(1)
        messages[0] = (messages[0][0], messages[0][1] + "\n" + second_msg[1])
        last_msg = messages[-1]
        messages.pop(-1)
        messages[1] = (messages[1][0], messages[1][1] + "\n" + last_msg[1])

    llm = build_llm(model)
    prompt_template = ChatPromptTemplate.from_messages(messages)
    llm_chain = (prompt_template | llm | JsonOutputParser())

    results[name][it]['intensity'] = {
        'model_error': None,
        'result': None
    }

    with open(src_file, 'r') as file:
        try:
            with get_openai_callback() as cb:
                result = llm_chain.invoke({'code': file.read()}, config={'callbacks': [ConsoleCallbackHandler()]})
                results[name][it]['intensity']['prompt_tokens'] = cb.prompt_tokens
                results[name][it]['intensity']['completion_tokens'] = cb.completion_tokens
                results[name][it]['intensity']['total_cost'] = cb.total_cost
                results[name][it]['intensity']['result'] = result['intensity']
        except Exception as e:
            results[name][it]['intensity']['model_error'] = str(e)


def extract_keys(model: str, name: str, src_file: str, it: int, results: dict) -> None:
    print(f"[IT #{it}] Extracting key data for {name}...")

    messages = [
        ("system", "Your task is to analyze C or C++ code of a stateful network function provided by the user. "
                   "The user will provide the header and the actual code of the network function. "
                   "You also act like a REST server, you only talk JSON, no natural language. "
                   "The user will provide the output format of the JSON."),
        ("system", "For each network function, the user will ask to determine the flow key of the NF. "
                   "A flow can be identified by one of the following fields, or a subset of them: "
                   "Source IP, Destination IP, Source Port, Destination Port, Protocol. "
                   "If the NF stores other types of data, they are not considered as part of a flow identifier."
                   "The extracted flow key should be contained in the following set of values: "
                   "['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']."),
        ("human", "Here is the C or C++ code of the network function to analyze:\n{code}"),
        ("human", "Provide a JSON containing a single key 'key' that contains the key of the NF "
                  "without any additional explanation."),
    ]
    if model_configurations[model]['type'] == 'google':
        # From https://python.langchain.com/docs/integrations/chat/google_generative_ai
        # For regular chat conversations, messages must follow the human/ai/human/ai alternating pattern.
        # You may not provide 2 AI or human messages in sequence
        second_msg = messages[1]
        messages.pop(1)
        messages[0] = (messages[0][0], messages[0][1] + "\n" + second_msg[1])
        last_msg = messages[-1]
        messages.pop(-1)
        messages[1] = (messages[1][0], messages[1][1] + "\n" + last_msg[1])

    llm = build_llm(model)
    prompt_template = ChatPromptTemplate.from_messages(messages)
    llm_chain = (prompt_template | llm | JsonOutputParser())

    results[name][it]['key'] = {
        'model_error': None,
        'result': None
    }

    with open(src_file, 'r') as file:
        try:
            with get_openai_callback() as cb:
                result = llm_chain.invoke({'code': file.read()}, config={'callbacks': [ConsoleCallbackHandler()]})
                results[name][it]['key']['prompt_tokens'] = cb.prompt_tokens
                results[name][it]['key']['completion_tokens'] = cb.completion_tokens
                results[name][it]['key']['total_cost'] = cb.total_cost
                results[name][it]['key']['result'] = result['key']
        except Exception as e:
            results[name][it]['key']['model_error'] = str(e)


def extract_state_size(model: str, name: str, src_file: str, it: int, results: dict) -> None:
    print(f"[IT #{it}] Extracting state size data for {name}...")

    messages = [
        ("system", "Your task is to analyze C or C++ code of a stateful network function provided by the user. "
                   "The user will provide the header and the actual code of the network function. "
                   "You also act like a REST server, you only talk JSON, no natural language. "
                   "The user will provide the output format of the JSON."),
        ("system", "For each network function, the user will ask to determine the size in bytes of the flow state "
                   "that the NF keeps track for each flow. If the state is stored in irregular memory "
                   "(using pointers), report it in the response."
                   "A flow can be identified by one of the following fields, or a subset of them: "
                   "Source IP, Destination IP, Source Port, Destination Port, Protocol. "
                   "If the NF stores other types of data, they are not considered as part of a flow."),
        ("human", "Here is the C or C++ code of the network function to analyze:\n{code}"),
        ("human", "Provide a JSON containing a single key 'pointer' that contains a boolean to report if the state"
                  " contains pointers (i.e., irregular memories)"
                  " without any additional explanation."
         ),
    ]
    if model_configurations[model]['type'] == 'google':
        # From https://python.langchain.com/docs/integrations/chat/google_generative_ai
        # For regular chat conversations, messages must follow the human/ai/human/ai alternating pattern.
        # You may not provide 2 AI or human messages in sequence
        second_msg = messages[1]
        messages.pop(1)
        messages[0] = (messages[0][0], messages[0][1] + "\n" + second_msg[1])
        last_msg = messages[-1]
        messages.pop(-1)
        messages[1] = (messages[1][0], messages[1][1] + "\n" + last_msg[1])

    llm = build_llm(model)
    prompt_template = ChatPromptTemplate.from_messages(messages)
    llm_chain = (prompt_template | llm | JsonOutputParser())

    results[name][it]['size'] = {
        'model_error': None,
        'result': None
    }

    with open(src_file, 'r') as file:
        try:
            with get_openai_callback() as cb:
                result = llm_chain.invoke({'code': file.read()})
                results[name][it]['size']['prompt_tokens'] = cb.prompt_tokens
                results[name][it]['size']['completion_tokens'] = cb.completion_tokens
                results[name][it]['size']['total_cost'] = cb.total_cost
                results[name][it]['size']['result'] = result['size']

                print(f"[IT #{it}] State size result for {name} is: ", results[name][it]['size']['result'])
        except Exception as e:
            results[name][it]['size']['model_error'] = str(e)


def extract_all(model: str, name: str, src_file: str, it: int, results: dict) -> None:
    print(f"[IT #{it}] Extracting merged data for {name}...")

    messages = [
        ("system", "Your task is to analyze C or C++ code of a stateful network function provided by the user. "
                   "The user will provide the header and the actual code of the network function. "
                   "You also act like a REST server, you only talk JSON, no natural language. "
                   "The user will provide the output format of the JSON."),
        ("system", "For each network function, the user will ask: 1. if the NF is stateful, "
                   "which means fetching and storing data per flow separately into a dedicated data structure; "
                   "2. how often the states are being updated, more specifically if states are being updated for "
                   "all packets of a flow or just once at the start of the flow, this is required only if the NF "
                   "is stateful; 3. to determine the flow key of the NF (the 5-tuple or a subset of it), this is "
                   "required only if the NF is stateful; 4. to determine the size in bytes of the flow state "
                   "that the NF keeps track for each flow. If the state is stored in irregular memory "
                   "(using pointers), report it in the response. This is required only if the NF "
                   "is stateful. A flow in a network function is identified by the 5-tuple, or a subset of them: "
                   "Source IP, Destination IP, Source Port, Destination Port, Protocol. "
                   "If the NF stores other types of data, it is not considered a flow."),
        ("human", "Here is the C or C++ code of the network function to analyze:\n{code}"),
        ("human", "Provide a JSON containing: 1. a key 'statefulness' that describes if the NF is stateful or not "
                  "without any additional explanation. Possible values are: 'stateful', 'stateless'; "
                  "2. a 'intensity' that describes the intensity of the NF without any additional explanation. "
                  "Possible values are: 'per-packet' when states are being updated per packet of a flow, and "
                  "'per-flow' when states are being updated occasionally for each flow. If the NF is stateless, put null; "
                  "3. a key 'key' that contains the key of the NF without any additional explanation. "
                  "The extracted flow key should be contained in the following set of values: "
                  "['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']. If the NF is stateless, put []; "
                  "4. a key 'pointer' that contains a boolean to report if the stored state per flow contains pointers "
                  "(i.e., irregular memories) without any additional explanation."
         ),
        #                  "4. a key 'size' that contains a tuple with the flow state size in bytes and a boolean to "
        #                  "report if the state is stored in irregular memory of the provided NF without "
        #                  "any additional explanation. If the NF is stateless or if you cannot infer the size, "
        #                  "put null instead of the tuple."),
    ]
    if model_configurations[model]['type'] == 'google':
        # From https://python.langchain.com/docs/integrations/chat/google_generative_ai
        # For regular chat conversations, messages must follow the human/ai/human/ai alternating pattern.
        # You may not provide 2 AI or human messages in sequence
        second_msg = messages[1]
        messages.pop(1)
        messages[0] = (messages[0][0], messages[0][1] + "\n" + second_msg[1])
        last_msg = messages[-1]
        messages.pop(-1)
        messages[1] = (messages[1][0], messages[1][1] + "\n" + last_msg[1])

    llm = build_llm(model)
    prompt_template = ChatPromptTemplate.from_messages(messages)
    llm_chain = (prompt_template | llm | JsonOutputParser())

    results[name][it]['merged'] = {
        'model_error': None,
        'result': None
    }

    with open(src_file, 'r') as file:
        try:
            with get_openai_callback() as cb:
                result = llm_chain.invoke({'code': file.read()})
                results[name][it]['merged']['prompt_tokens'] = cb.prompt_tokens
                results[name][it]['merged']['completion_tokens'] = cb.completion_tokens
                results[name][it]['merged']['total_cost'] = cb.total_cost
                results[name][it]['merged']['result'] = result

                print(f"[IT #{it}] Merged result for {name} is: ", results[name][it]['merged']['result'])
        except Exception as e:
            results[name][it]['merged']['model_error'] = str(e)


def extract_complexity(model: str, pairs: list, it: int, results: dict) -> None:
    messages = [
        ("system", "Your task is to analyze C or C++ code of two stateful network functions provided by the user. "
                   "The user will provide the header and the actual code of the network function. "
                   "You also act like a REST server, you only talk JSON, no natural language. "
                   "The user will provide the output format of the JSON."),
        ("system", "The user will ask how to estimate which one of the two NFs requires more processing per packet."),
        ("human", "Here is the C or C++ code of the two network functions to analyze:\n{code1}\n{code2}"),
        ("human", "Provide a JSON containing a single key 'complexity' that contains the name of the NF "
                  "that requires more processing per packet or 'ANY' if both NFs require almost the same "
                  "amount of processing per packet, without any additional explanation."),
    ]
    if model_configurations[model]['type'] == 'google':
        # From https://python.langchain.com/docs/integrations/chat/google_generative_ai
        # For regular chat conversations, messages must follow the human/ai/human/ai alternating pattern.
        # You may not provide 2 AI or human messages in sequence
        second_msg = messages[1]
        messages.pop(1)
        messages[0] = (messages[0][0], messages[0][1] + "\n" + second_msg[1])
        last_msg = messages[-1]
        messages.pop(-1)
        messages[1] = (messages[1][0], messages[1][1] + "\n" + last_msg[1])

    llm = build_llm(model)
    prompt_template = ChatPromptTemplate.from_messages(messages)
    llm_chain = (prompt_template | llm | JsonOutputParser())

    for (name1, src_path1), (name2, src_path2) in pairs:
        print(f"[IT #{it}] Extracting complexity data for pair {name1} and {name2}...")

        with open(src_path1, 'r') as file:
            code1 = file.read()
        with open(src_path2, 'r') as file:
            code2 = file.read()

        results['complexity'][it][f"({name1},{name2})"] = {
            'model_error': None,
            'result': None
        }

        try:
            with get_openai_callback() as cb:
                result = llm_chain.invoke({'code1': code1, 'code2': code2})
                results['complexity'][it][f"({name1},{name2})"]['prompt_tokens'] = cb.prompt_tokens
                results['complexity'][it][f"({name1},{name2})"]['completion_tokens'] = cb.completion_tokens
                results['complexity'][it][f"({name1},{name2})"]['total_cost'] = cb.total_cost
                results['complexity'][it][f"({name1},{name2})"]['result'] = result['complexity']
        except Exception as e:
            results['complexity'][it][f"({name1},{name2})"]['model_error'] = str(e)

        print(f"[IT #{it}] Result for {name1} and {name2}: ", results['complexity'][it][f"({name1},{name2})"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(model_configurations.keys()),
                        default="gpt-4-turbo", required=False)
    parser.add_argument('--filter', type=str, required=False, default="IPortScanDetector")
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--source_path', type=str, required=False, default="./extracted_nfs/")
    parser.add_argument('--results_path', type=str, required=False, default="./llm_results/")

    return parser.parse_args()


def main(args: argparse.Namespace):
    curr_path = os.path.dirname(__file__)

    source_path = os.path.abspath(os.path.join(curr_path, args.source_path))
    results_path = os.path.abspath(os.path.join(curr_path, args.results_path))
    module_filter = [x.strip() for x in args.filter.split(',')] if args.filter else []

    merged_files = find_merged_files(source_path)

    results = {'complexity': {}}
    for it in range(1, args.iterations + 1):
        for name, merged_file in merged_files.items():
            if len(module_filter) > 0 and name not in module_filter:
                continue

            if name not in results:
                results[name] = {}

            if it not in results:
                results[name][it] = {}

            extract_statefulness(args.model, name, merged_file, it, results)
            extract_read_write_intensity(args.model, name, merged_file, it, results)
            extract_keys(args.model, name, merged_file, it, results)
            extract_state_size(args.model, name, merged_file, it, results)
            extract_all(args.model, name, merged_file, it, results)

        results['complexity'][it] = {}
        pairs = itertools.combinations(filter(
            lambda x: not module_filter or len(module_filter) > 0 and x[0] in module_filter,
            [x for x in merged_files.items() if x[0] in stateful_modules]
        ), 2)
    #        extract_complexity(args.model, pairs, it, results)

    os.makedirs(results_path, exist_ok=True)

    results_time = time.strftime("%Y%m%d-%H%M%S")
    filename = f"result-{args.model}-{results_time}.json"

    with open(os.path.join(results_path, filename), 'w') as result_file:
        result_file.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    main(parse_args())
