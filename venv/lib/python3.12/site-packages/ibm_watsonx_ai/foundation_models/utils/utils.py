#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import asyncio
from enum import Enum
import types
import copy
import functools
from string import Formatter
from typing import (
    TYPE_CHECKING,
    Any,
    Sequence,
    cast,
    Mapping,
    Type,
    Generator,
    Callable,
)
from dataclasses import dataclass, KW_ONLY, asdict
from json import loads as json_loads
from warnings import warn, simplefilter, catch_warnings
from pprint import pprint
import threading

from ibm_watsonx_ai._wrappers import requests
from ibm_watsonx_ai.helpers import DataConnection
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    InvalidMultipleArguments,
    InvalidValue,
)
from ibm_watsonx_ai.utils import next_resource_generator, get_user_agent_header
from ibm_watsonx_ai.utils.autoai.utils import load_file_from_file_system_nonautoai
from ibm_watsonx_ai.utils.autoai.enums import DataConnectionTypes
from ibm_watsonx_ai.lifecycle import SpecStates

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from types import TracebackType, FrameType
    from asyncio import AbstractEventLoop
    from concurrent.futures import Future


@dataclass
class PromptTuningParams:
    base_model: dict
    _: KW_ONLY
    accumulate_steps: int | None = None
    batch_size: int | None = None
    init_method: str | None = None
    init_text: str | None = None
    learning_rate: float | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    num_epochs: int | None = None
    task_id: str | None = None
    tuning_type: str | None = None
    verbalizer: str | None = None

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class FineTuningParams:
    base_model: dict
    task_id: str
    _: KW_ONLY
    num_epochs: int | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    max_seq_length: int | None = None
    accumulate_steps: int | None = None
    verbalizer: str | None = None
    response_template: str | None = None
    gpu: dict | None = None
    peft_parameters: dict | None = None
    gradient_checkpointing: bool | None = None

    def to_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _get_foundation_models_spec(
    url: str, operation_name: str, additional_params: dict | None = None
) -> dict:
    params = {"version": "2023-09-30"}
    if additional_params:
        params.update(additional_params)
    response = requests.get(
        url, params=params, headers={"User-Agent": get_user_agent_header()}
    )
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        raise WMLClientError(
            Messages.get_message(message_id="fm_prompt_tuning_no_foundation_models"),
            logg_messages=False,
        )
    else:
        msg = f"{operation_name} failed. Reason: {response.text}"
        raise WMLClientError(msg)


def get_model_specs(url: str, model_id: str | None = None) -> dict:
    """
    Retrieve the list of deployed foundation models specifications.

    **Decrecated:** From ``ibm_watsonx_ai`` 1.0, the `get_model_specs()` function is deprecated, use the `client.foundation_models.get_model_specs()` function instead.

    :param url: URL of the environment
    :type url: str

    :param model_id: ID of the model, defaults to None (all models specs are returned).
    :type model_id: Optional[str, ModelTypes], optional

    :return: list of deployed foundation model specs
    :rtype: dict

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai.foundation_models import get_model_specs

        # GET ALL MODEL SPECS
        get_model_specs(
            url="https://us-south.ml.cloud.ibm.com"
            )

        # GET MODEL SPECS BY MODEL_ID
        get_model_specs(
            url="https://us-south.ml.cloud.ibm.com",
            model_id="google/flan-ul2"
            )
    """

    get_model_specs_deprecated_warning = (
        "`get_model_specs()` function is deprecated from 1.0, "
        "please use `client.foundation_models.get_model_specs()` function instead."
    )
    warn(get_model_specs_deprecated_warning, category=DeprecationWarning)

    try:
        if model_id:
            if isinstance(model_id, Enum):
                model_id = model_id.value
            try:
                return [
                    res
                    for res in _get_foundation_models_spec(
                        f"{url}/ml/v1/foundation_model_specs",
                        "Get available foundation models",
                    )["resources"]
                    if res["model_id"] == model_id
                ][0]
            except WMLClientError:  # Remove on CPD 5.0 release
                return [
                    res
                    for res in _get_foundation_models_spec(
                        f"{url}/ml/v1-beta/foundation_model_specs",
                        "Get available foundation models",
                    )["resources"]
                    if res["model_id"] == model_id
                ][0]
        else:
            try:
                return _get_foundation_models_spec(
                    f"{url}/ml/v1/foundation_model_specs",
                    "Get available foundation models",
                )
            except WMLClientError:  # Remove on CPD 5.0 release
                return _get_foundation_models_spec(
                    f"{url}/ml/v1-beta/foundation_model_specs",
                    "Get available foundation models",
                )
    except WMLClientError as e:
        raise WMLClientError(
            Messages.get_message(url, message_id="fm_prompt_tuning_no_model_specs"),
            e.reason,
        )


def get_custom_model_specs(
    credentials: dict | None = None,
    api_client: APIClient | None = None,
    model_id: str | None = None,
    limit: int = 100,
    asynchronous: bool = False,
    get_all: bool = False,
    verify: str | bool | None = None,
) -> dict | Generator:
    """Get details on available custom model(s) as dict or as generator (``asynchronous``).
    If ``asynchronous`` or ``get_all`` is set, then ``model_id`` is ignored.

    **Decrecated:** From ``ibm_watsonx_ai`` 1.0, the `get_custom_model_specs()` function is deprecated, use `client.foundation_models.get_custom_model_specs()` function instead.

    :param credentials: credentials to watsonx.ai instance
    :type credentials: dict, optional

    :param api_client: API client to connect to service
    :type api_client: APIClient, optional

    :param model_id: Id of the model, defaults to None (all models specs are returned).
    :type model_id: str, optional

    :param limit:  limit number of fetched records. Possible values: 1 ≤ value ≤ 200, default value: 100
    :type limit: int, optional

    :param asynchronous:  if True, it will work as a generator
    :type asynchronous: bool, optional

    :param get_all:  if True, it will get all entries in 'limited' chunks
    :type get_all: bool, optional

    :param verify: You can pass one of following as verify:

        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * `True` - default path to truststore will be taken
        * `False` - no verification will be made
    :type verify: bool or str, optional

    :return: details of supported custom models
    :rtype: dict

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai.foundation_models import get_custom_model_specs

        get_custom_models_spec(api_client=client)
        get_custom_models_spec(credentials=credentials)
        get_custom_models_spec(api_client=client, model_id='mistralai/Mistral-7B-Instruct-v0.2')
        get_custom_models_spec(api_client=client, limit=20)
        get_custom_models_spec(api_client=client, limit=20, get_all=True)
        for spec in get_custom_model_specs(api_client=client, limit=20, asynchronous=True, get_all=True):
            print(spec, end="")

    """

    get_custom_model_specs_deprecated_warning = (
        "`get_custom_model_specs()` function is deprecated from 1.0, "
        "please use `client.foundation_models.get_custom_model_specs()` function instead."
    )
    warn(get_custom_model_specs_deprecated_warning, category=DeprecationWarning)

    model_usage_warning = (
        "Model needs to be first stored via client.repository.store_model(model_id, meta_props=metadata) "
        "and deployed via client.deployments.create(asset_id, metadata) to be used."
    )
    warn(model_usage_warning)

    if credentials is None and api_client is None:
        raise InvalidMultipleArguments(
            params_names_list=["credentials", "api_client"],
            reason="None of the arguments were provided.",
        )
    elif credentials:
        from ibm_watsonx_ai import APIClient

        client = APIClient(credentials, verify=verify)
    else:
        client = api_client  # type: ignore[assignment]

    url = client.credentials.url

    params = client._params(skip_for_create=True, skip_userfs=True)
    if limit < 1 or limit > 200:
        raise InvalidValue(
            value_name="limit",
            reason=f"The given value {limit} is not in the range <1, 200>",
        )
    else:
        params.update({"limit": limit})

    url = cast(str, url)
    if asynchronous or get_all:
        resource_generator = next_resource_generator(
            client,
            url=url,
            href="ml/v4/custom_foundation_models",
            params=params,
            _all=get_all,
        )

        if asynchronous:
            return resource_generator

        resources = []
        for entry in resource_generator:
            resources.extend(entry["resources"])
        return {"resources": resources}

    response = requests.get(
        f"{url}/ml/v4/custom_foundation_models",
        params=params,
        headers=client._get_headers(),
    )
    if response.status_code == 200:
        if model_id:
            resources = [
                res
                for res in response.json()["resources"]
                if res["model_id"] == model_id
            ]
            return resources[0] if resources else {}
        else:
            return response.json()
    elif response.status_code == 404:
        raise WMLClientError(
            Messages.get_message(url, message_id="custom_models_no_model_specs"), url
        )
    else:
        msg = f"Getting failed. Reason: {response.text}"
        raise WMLClientError(msg)


def get_model_lifecycle(url: str, model_id: str) -> list | None:
    """
    Retrieve the list of model lifecycle data.

    **Decrecated:** From ``ibm_watsonx_ai`` 1.0, the `get_model_lifecycle()` function is deprecated, use `client.foundation_models.get_model_lifecycle()` function instead.

    :param url: URL of environment
    :type url: str

    :param model_id: the type of model to use
    :type model_id: str

    :return: list of deployed foundation model lifecycle data
    :rtype: list

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai.foundation_models import get_model_lifecycle

        get_model_lifecycle(
            url="https://us-south.ml.cloud.ibm.com",
            model_id="ibm/granite-13b-instruct-v2"
            )
    """

    get_model_lifecycle_deprecated_warning = (
        "`get_model_lifecycle()` function is deprecated from 1.0, "
        "please use `client.foundation_models.get_model_lifecycle()` function instead."
    )
    warn(get_model_lifecycle_deprecated_warning, category=DeprecationWarning)

    model_specs = get_model_specs(url=url)
    model_spec = next(
        (
            model_metadata
            for model_metadata in model_specs.get("resources", [])
            if model_metadata.get("model_id") == model_id
        ),
        None,
    )
    return model_spec.get("lifecycle") if model_spec is not None else None


def _check_model_state(
    client: APIClient,
    model_id: str,
    tech_preview: bool = False,
    model_specs: dict | None = None,
) -> None:
    default_warning_template = (
        "Model '{model_id}' is in {state} state from {start_date} until {withdrawn_start_date}. "
        "IDs of alternative models: {alternative_model_ids}. "
        "Further details: https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-model-lifecycle.html?context=wx&audience=wdp"
    )

    if model_specs is not None:
        lifecycle = None
        for model_spec in model_specs["resources"]:
            if model_spec["model_id"] == model_id:
                lifecycle = model_spec.get("lifecycle")
                break
    elif client._use_fm_ga_api:
        lifecycle = client.foundation_models.get_model_lifecycle(
            model_id, tech_preview=tech_preview
        )
    else:
        with catch_warnings():
            simplefilter("ignore", category=DeprecationWarning)
            lifecycle = get_model_lifecycle(client.credentials.url, model_id)  # type: ignore[arg-type]

    modes_list = [ids.get("id") for ids in (lifecycle or [])]
    deprecated_or_constricted_warning_template_cpd = (
        "Model '{model_id}' is in {state} state. "
        "IDs of alternative models: {alternative_model_ids}. "
        "Further details: https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-model-lifecycle.html?context=wx&audience=wdp"
    )

    if lifecycle and SpecStates.DEPRECATED.value in modes_list:
        model_lifecycle = next(
            (el for el in lifecycle if el.get("id") == SpecStates.DEPRECATED.value),
            None,
        )
        model_lifecycle = cast(dict, model_lifecycle)
        if model_lifecycle.get("since_version"):
            model_state_warning = deprecated_or_constricted_warning_template_cpd.format(
                model_id=model_id,
                state=(model_lifecycle.get("label") or SpecStates.DEPRECATED.value),
                alternative_model_ids=", ".join(
                    model_lifecycle.get("alternative_model_ids", ["None"])
                ),
            )
        else:
            model_state_warning = default_warning_template.format(
                model_id=model_id,
                state=(model_lifecycle.get("label") or SpecStates.DEPRECATED.value),
                start_date=model_lifecycle.get("start_date"),
                withdrawn_start_date=next(
                    (
                        el.get("start_date")
                        for el in lifecycle
                        if el.get("id") == SpecStates.WITHDRAWN.value
                    ),
                    None,
                ),
                alternative_model_ids=", ".join(
                    model_lifecycle.get("alternative_model_ids", ["None"])
                ),
            )

        warn(model_state_warning, category=LifecycleWarning)

    elif lifecycle and SpecStates.CONSTRICTED.value in modes_list:
        model_lifecycle = next(
            (el for el in lifecycle if el.get("id") == SpecStates.CONSTRICTED.value),
            None,
        )
        model_lifecycle = cast(dict, model_lifecycle)
        if model_lifecycle.get("since_version"):
            model_state_warning = deprecated_or_constricted_warning_template_cpd.format(
                model_id=model_id,
                state=(model_lifecycle.get("label") or SpecStates.CONSTRICTED.value),
                alternative_model_ids=", ".join(
                    model_lifecycle.get("alternative_model_ids", ["None"])
                ),
            )
        else:
            model_state_warning = default_warning_template.format(
                model_id=model_id,
                state=(model_lifecycle.get("label") or SpecStates.CONSTRICTED.value),
                start_date=model_lifecycle.get("start_date"),
                withdrawn_start_date=next(
                    (
                        el.get("start_date")
                        for el in lifecycle
                        if el.get("id") == SpecStates.WITHDRAWN.value
                    ),
                    None,
                ),
                alternative_model_ids=", ".join(
                    model_lifecycle.get("alternative_model_ids", ["None"])
                ),
            )

        warn(model_state_warning, category=LifecycleWarning)


def get_model_specs_with_prompt_tuning_support(url: str) -> dict:
    """
    Query the details of the deployed foundation models with prompt tuning support.

    **Decrecated:** From ``ibm_watsonx_ai`` 1.0, the `get_model_specs_with_prompt_tuning_support()` function is deprecated, use the `client.foundation_models.get_model_specs_with_prompt_tuning_support()` function instead.

    :param url: URL of environment
    :type url: str

    :return: list of deployed foundation model specs with prompt tuning support
    :rtype: dict

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai.foundation_models import get_model_specs_with_prompt_tuning_support

        get_model_specs_with_prompt_tuning_support(
            url="https://us-south.ml.cloud.ibm.com"
            )
    """

    get_model_specs_with_prompt_deprecated_warning = (
        "`get_model_specs_with_prompt_tuning_support()` function is deprecated from 1.0, "
        "please use `client.foundation_models.get_model_specs_with_prompt_tuning_support()` function instead."
    )
    warn(get_model_specs_with_prompt_deprecated_warning, category=DeprecationWarning)

    try:
        try:
            return _get_foundation_models_spec(
                url=f"{url}/ml/v1/foundation_model_specs",
                operation_name="Get available foundation models",
                additional_params={"filters": "function_prompt_tune_trainable"},
            )
        except WMLClientError:  # Remove on CPD 5.0 release
            return _get_foundation_models_spec(
                url=f"{url}/ml/v1-beta/foundation_model_specs",
                operation_name="Get available foundation models",
                additional_params={"filters": "function_prompt_tune_trainable"},
            )
    except WMLClientError as e:
        raise WMLClientError(
            Messages.get_message(url, message_id="fm_prompt_tuning_no_model_specs"),
            e.reason,
        )


def get_supported_tasks(url: str) -> dict:
    """
    Retrieves a list of tasks that are supported by the foundation models.

    :param url: URL of the environment
    :type url: str

    :return: list of tasks that are supported by the foundation models
    :rtype: dict

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai.foundation_models import get_supported_tasks

        get_supported_tasks(
            url="https://us-south.ml.cloud.ibm.com"
            )
    """
    try:
        try:
            return _get_foundation_models_spec(
                f"{url}/ml/v1/foundation_model_tasks",
                "Get tasks that are supported by the foundation models.",
            )
        except WMLClientError:  # Remove on CPD 5.0 release
            return _get_foundation_models_spec(
                f"{url}/ml/v1-beta/foundation_model_tasks",
                "Get tasks that are supported by the foundation models.",
            )
    except WMLClientError as e:
        raise WMLClientError(
            Messages.get_message(url, message_id="fm_prompt_tuning_no_supported_tasks"),
            e.reason,
        )


def get_all_supported_tasks_dict(
    url: str = "https://us-south.ml.cloud.ibm.com",
) -> dict:
    tasks_dict = dict()
    for task_spec in get_supported_tasks(url).get("resources", []):
        tasks_dict[task_spec["label"].replace("-", "_").replace(" ", "_").upper()] = (
            task_spec["task_id"]
        )
    return tasks_dict


def load_request_json(
    run_id: str,
    api_client: APIClient,
    run_params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    # note: backward compatibility
    if (wml_client := kwargs.get("wml_client")) is None and api_client is None:
        raise WMLClientError("No `api_client` provided")
    elif wml_client is not None:
        if api_client is None:
            api_client = wml_client

        wml_client_parameter_deprecated_warning = (
            "`wml_client` parameter is deprecated and will be removed in future. "
            "Instead, please use `api_client`."
        )
        warn(wml_client_parameter_deprecated_warning, category=DeprecationWarning)
    # --- end note
    if run_params is None:
        run_params = api_client.training.get_details(run_id)

    model_request_path = (
        run_params["entity"]
        .get("results_reference", {})
        .get("location", {})
        .get("model_request_path")
    )

    if model_request_path is None:
        raise WMLClientError(
            "Missing model_request_path in run_params. Verify if the training run has been completed."
        )

    if api_client.CLOUD_PLATFORM_SPACES:
        results_reference = DataConnection._from_dict(
            run_params["entity"]["results_reference"]
        )

        if run_params["entity"]["results_reference"]["type"] == DataConnectionTypes.CA:
            results_reference.location.file_name = model_request_path  # type: ignore
        else:
            results_reference.location.path = model_request_path  # type: ignore[union-attr]

        results_reference.set_client(api_client)
        request_json_bytes = results_reference.read(raw=True, binary=True)

        # download from cos

    elif api_client.CPD_version >= 4.8:
        asset_parts = model_request_path.split("/")
        model_request_asset_url = "/".join(
            asset_parts[asset_parts.index("assets") + 1 :]
        )
        request_json_bytes = load_file_from_file_system_nonautoai(
            api_client=api_client, file_path=model_request_asset_url
        ).read()
    else:
        raise WMLClientError("Unsupported environment for this action")

    request_json_bytes = cast(bytes, request_json_bytes)
    return json_loads(request_json_bytes.decode())


def is_training_prompt_tuning(
    training_id: str | None = None, api_client: APIClient | None = None, **kwargs: Any
) -> bool:
    """Returns True if training_id is connected to prompt tuning"""
    # note: backward compatibility
    if (wml_client := kwargs.get("wml_client")) is None and api_client is None:
        raise WMLClientError("No `api_client` provided")
    elif wml_client is not None:
        if api_client is None:
            api_client = wml_client

        wml_client_parameter_deprecated_warning = (
            "`wml_client` parameter is deprecated and will be removed in future. "
            "Instead, please use `api_client`."
        )
        warn(wml_client_parameter_deprecated_warning, category=DeprecationWarning)
    # --- end note
    if training_id is None:
        return False
    run_params = api_client.training.get_details(training_id=training_id)  # type: ignore[union-attr]
    return bool(run_params["entity"].get("prompt_tuning"))


class TemplateFormatter(Formatter):
    def check_unused_args(
        self, used_args: set[(int | str)], args: Sequence, kwargs: Mapping[str, Any]
    ) -> None:
        """Check for unused args."""
        extra_args = set(kwargs).difference(used_args)
        if extra_args:
            raise KeyError(extra_args)


class HAPDetectionWarning(UserWarning): ...


class PIIDetectionWarning(UserWarning): ...


class LifecycleWarning(UserWarning): ...


class WatsonxLLMDeprecationWarning(UserWarning): ...


def _raise_watsonxllm_deprecation_warning() -> None:
    watsonx_llm_deprecated_warning = (
        "ibm_watsonx_ai.foundation_models.extensions.langchain.WatsonxLLM "
        "is deprecated and will not be supported in the future. "
        "Please import from langchain-ibm instead.\n"
        "To install langchain-ibm run `pip install -U langchain-ibm`."
    )
    warn(watsonx_llm_deprecated_warning, category=WatsonxLLMDeprecationWarning, stacklevel=2)  # fmt: skip


def get_embedding_model_specs(url: str) -> dict:
    """
    **Decrecated:** From ``ibm_watsonx_ai`` 1.0, the `get_embedding_model_specs()` function is deprecated, use the `client.foundation_models.get_embeddings_model_specs()` function instead.
    """
    get_embedding_model_specs_deprecated_warning = (
        "`get_embedding_model_specs()` function is deprecated from 1.0, "
        "please use `client.foundation_models.get_embeddings_model_specs()` function instead."
    )
    warn(get_embedding_model_specs_deprecated_warning, category=DeprecationWarning)

    return _get_foundation_models_spec(
        url=f"{url}/ml/v1/foundation_model_specs",
        operation_name="Get available embedding models",
        additional_params={"filters": "function_embedding"},
    )


def _copy_function(func: Callable) -> Callable:
    """Custom copy of function"""

    if not isinstance(func, types.FunctionType):
        raise TypeError("The provided object is not of a callable function type.")

    new_func = types.FunctionType(
        func.__code__,
        func.__globals__,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=func.__closure__,
    )

    new_func.__dict__.update(copy.deepcopy(func.__dict__))
    new_func.__annotations__ = func.__annotations__.copy()
    new_func.__doc__ = func.__doc__

    return new_func


def _is_fine_tuning_endpoint_available(api_client: APIClient) -> bool:

    try:
        url = api_client.service_instance._href_definitions.get_fine_tunings_href()

        response_fine_tuning_api = api_client._session.get(
            url=f"{url}?limit=1",
            params=api_client._params(),
            headers=api_client._get_headers(),
        )
        return response_fine_tuning_api.status_code == 200

    except:
        return False
