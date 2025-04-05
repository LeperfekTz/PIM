#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from typing import Any, Type, TypeVar, get_origin, get_args
from tabulate import tabulate
from enum import Enum

from ibm_watsonx_ai.utils.utils import StrEnum
from dataclasses import dataclass, is_dataclass, fields


T = TypeVar("T", bound="BaseSchema")


@dataclass
class BaseSchema:

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> "BaseSchema":
        kwargs = {}
        for field in fields(cls):
            field_name = field.name
            field_type = field.type
            if field_name in data:
                value = data[field_name]
                origin = get_origin(field_type)
                if origin is not None and issubclass(origin, BaseSchema):
                    if hasattr(origin, "from_dict"):
                        value = origin.from_dict(value)
                kwargs[field_name] = value
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        def unpack(
            value: Enum | list[Any] | Any,
        ) -> int | dict[str, Any] | list[Any] | Any:
            if isinstance(value, Enum):
                return value.value
            elif is_dataclass(value):
                return {
                    k: unpack(v)
                    for k, v in value.__dict__.items()
                    if v is not None and not k.startswith("_")
                }
            elif isinstance(value, list):
                return [unpack(v) for v in value]
            else:
                return value

        return {
            k: unpack(v)
            for k, v in self.__dict__.items()
            if v is not None and not k.startswith("_")
        }

    @classmethod
    def show(cls) -> None:
        """Displays a table with the parameter name, type, and example value."""
        sample_params = cls.get_sample_params()
        table_data = []
        for field in fields(cls):
            field_name = field.name
            field_type = field.type
            origin = get_origin(field_type) or field_type
            args = get_args(field_type)
            if args:
                display_type = f"{', '.join(arg.__name__ if hasattr(arg, '__name__') else str(arg) for arg in args)}"
            else:
                display_type = (
                    origin.__name__ if hasattr(origin, "__name__") else str(origin)
                )

            example_value = sample_params.get(field_name, "N/A")
            table_data.append([field_name, display_type, example_value])

        print(
            tabulate(
                table_data,
                headers=["PARAMETER", "TYPE", "EXAMPLE VALUE"],
                tablefmt="grid",
            )
        )

    @classmethod
    def get_sample_params(cls) -> dict[str, Any]:
        """Override this method in subclasses to provide example values for parameters."""
        return {}


##############
#  TEXT-GEN  #
##############


class TextGenDecodingMethod(StrEnum):
    GREEDY = "greedy"
    SAMPLE = "sample"


@dataclass
class TextGenLengthPenalty(BaseSchema):
    decay_factor: float | None = None
    start_index: int | None = None

    @classmethod
    def get_sample_params(cls) -> dict[str, Any]:
        """Provide example values for TextGenLengthPenalty."""
        return {
            "decay_factor": 2.5,
            "start_index": 5,
        }


@dataclass
class ReturnOptionProperties(BaseSchema):
    input_text: bool | None = None
    generated_tokens: bool | None = None
    input_tokens: bool | None = None
    token_logprobs: bool | None = None
    token_ranks: bool | None = None
    top_n_tokens: bool | None = None

    @classmethod
    def get_sample_params(cls) -> dict[str, Any]:
        """Provide example values for ReturnOptionProperties."""
        return {
            "input_text": True,
            "generated_tokens": True,
            "input_tokens": True,
            "token_logprobs": True,
            "token_ranks": False,
            "top_n_tokens": False,
        }


@dataclass
class TextGenParameters(BaseSchema):
    decoding_method: str | TextGenDecodingMethod | None = None
    length_penalty: dict | TextGenLengthPenalty | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    random_seed: int | None = None
    repetition_penalty: float | None = None
    min_new_tokens: int | None = None
    max_new_tokens: int | None = None
    stop_sequences: list[str] | None = None
    time_limit: int | None = None
    truncate_input_tokens: int | None = None
    return_options: dict | ReturnOptionProperties | None = None
    include_stop_sequence: bool | None = None
    prompt_variables: dict | None = None

    @classmethod
    def get_sample_params(cls) -> dict[str, Any]:
        """Provide example values for TextChatParameters."""
        return {
            "decoding_method": list(TextGenDecodingMethod)[1].value,
            "length_penalty": TextGenLengthPenalty.get_sample_params(),
            "temperature": 0.5,
            "top_p": 0.2,
            "top_k": 1,
            "random_seed": 33,
            "repetition_penalty": 2,
            "min_new_tokens": 50,
            "max_new_tokens": 1000,
            "stop_sequences": 200,
            "time_limit": 600000,
            "truncate_input_tokens": 200,
            "return_options": ReturnOptionProperties.get_sample_params(),
            "include_stop_sequence": True,
            "prompt_variables": {"doc_type": "emails", "entity_name": "Golden Retail"},
        }


###############
#  TEXT-CHAT  #
###############


class TextChatResponseFormatType(StrEnum):
    JSON_OBJECT = "json_object"


@dataclass
class TextChatResponseFormat(BaseSchema):
    type: str | TextChatResponseFormatType | None = None

    @classmethod
    def get_sample_params(cls) -> dict[str, Any]:
        """Provide example values for TextChatResponseFormat."""
        return {"type": list(TextChatResponseFormatType)[0].value}


@dataclass
class TextChatParameters(BaseSchema):
    frequency_penalty: float | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    presence_penalty: float | None = None
    response_format: dict | TextChatResponseFormat | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    time_limit: int | None = None
    top_p: float | None = None
    n: int | None = None
    logit_bias: dict | None = None
    seed: int | None = None
    stop: list[str] | None = None

    @classmethod
    def get_sample_params(cls) -> dict[str, Any]:
        """Provide example values for TextChatParameters."""
        return {
            "frequency_penalty": 0.5,
            "logprobs": True,
            "top_logprobs": 3,
            "presence_penalty": 0.3,
            "response_format": TextChatResponseFormat.get_sample_params(),
            "temperature": 0.7,
            "max_tokens": 100,
            "time_limit": 600000,
            "top_p": 0.9,
            "n": 1,
            "logit_bias": {"1003": -100, "1004": -100},
            "seed": 41,
            "stop": ["this", "the"],
        }


############
#  RERANK  #
############


@dataclass
class RerankReturnOptions(BaseSchema):
    top_n: int | None = None
    inputs: bool | None = None
    query: bool | None = None

    @classmethod
    def get_sample_params(cls) -> dict[str, Any]:
        """Provide example values for RerankReturnOptions."""
        return {"top_n": 1, "inputs": False, "query": False}


@dataclass
class RerankParameters(BaseSchema):
    truncate_input_tokens: int | None = None
    return_options: dict | RerankReturnOptions | None = None

    @classmethod
    def get_sample_params(cls) -> dict[str, Any]:
        """Provide example values for RerankParameters."""
        return {
            "truncate_input_tokens": 100,
            "return_options": RerankReturnOptions.get_sample_params(),
        }


#################
#  TIME SERIES  #
#################


@dataclass
class TSForecastParameters(BaseSchema):
    r"""
    :param timestamp_column: A valid column in the data that should be treated as the timestamp.  if using calendar dates (simple integer time offsets are also allowed), users should consider using a format such as ISO 8601 that includes a UTC offset (e.g., '2024-10-18T01:09:21.454746+00:00'). This will avoid potential issues such as duplicate dates appearing due to daylight savings change overs. There are many date formats in existence and inferring the correct one can be a challenge so please do consider adhering to ISO 8601.
    :type timestamp_column: str

    :param prediction_length: The prediction length for the forecast. The service will return this many periods beyond the last timestamp in the inference data payload. If specified, prediction_length must be an integer >=1 and no more than the model default prediction length. When omitted the model default prediction_length will be used.
    :type prediction_length: int, optional

    :param id_columns: Columns that define a unique key for time series. This is similar to a compound primary key in a database table.
    :type id_columns: list[str], optional

    :param freq: A frequency indicator for the given timestamp_column. See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases for a description of the allowed values. If not provided, we will attempt to infer it from the data. Possible values: 0 ≤ length ≤ 100, Value must match regular expression ^\d+(B|D|W|M|Q|Y|h|min|s|ms|us|ns)$|^\s*$
    :type freq: str, optional

    :param target_columns: An array of column headings which constitute the target variables. These are the data that will be forecasted.
    :type target_columns: list[str], optional

    """

    timestamp_column: str
    prediction_length: int | None = None
    id_columns: list[str] | None = None
    freq: str | None = None
    target_columns: list[str] | None = None
    observable_columns: list[str] | None = None
    control_columns: list[str] | None = None
    conditional_columns: list[str] | None = None
    static_categorical_columns: list[str] | None = None

    @classmethod
    def get_sample_params(cls) -> dict[str, Any]:
        """Provide example values for TSForecastParameters."""
        return {
            "prediction_length": 10,
            "timestamp_column": "date",
            "id_columns": ["id1"],
            "freq": "D",
            "target_columns": ["col1", "col2"],
        }


#################
#  FINE TUNING  #
#################


@dataclass
class PeftParameters(BaseSchema):

    type: str
    rank: int | None = None
    target_modules: list | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None

    @classmethod
    def get_sample_params(cls) -> dict[str, Any]:
        """Provide example values for PeftParameters."""
        return {
            "type": "lora",
            "rank": 8,
            "target_modules": ["all-linear"],
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        }
