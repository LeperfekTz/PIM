#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING, TypeAlias
from warnings import catch_warnings, simplefilter


from ibm_watsonx_ai.foundation_models.prompt_tuner import PromptTuner
from ibm_watsonx_ai.foundation_models.fine_tuner import FineTuner
from ibm_watsonx_ai.foundation_models.ilab_tuner import ILabTuner
from ibm_watsonx_ai.foundation_models.utils.enums import (
    PromptTuningTypes,
    PromptTuningInitMethods,
    TuneExperimentTasks,
)
from ibm_watsonx_ai.foundation_models.utils.utils import (
    _check_model_state,
    get_model_specs_with_prompt_tuning_support,
)
from ibm_watsonx_ai.experiment.base_experiment import BaseExperiment
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai import APIClient

from .tune_runs import TuneRuns

if TYPE_CHECKING:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models.schema import PeftParameters

EnumLike: TypeAlias = EnumMeta | Enum


class TuneExperiment(BaseExperiment):
    """The TuneExperiment class for tuning models with prompts.

    :param credentials: credentials for the Watson Machine Learning instance
    :type credentials: Credentials or dict

    :param project_id: ID of the Watson Studio project
    :type project_id: str, optional

    :param space_id: ID of the Watson Studio space
    :type space_id: str, optional

    :param verify: You can pass one of following as verify:

        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * `True` - default path to truststore will be taken
        * `False` - no verification will be made
    :type verify: bool or str, optional

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.experiment import TuneExperiment

        experiment = TuneExperiment(
            credentials=Credentials(...),
            project_id="...",
            space_id="...")
    """

    def __init__(
        self,
        credentials: Credentials | dict[str, str],
        project_id: str | None = None,
        space_id: str | None = None,
        verify: str | bool | None = None,
    ) -> None:

        self.client = APIClient(credentials, verify=verify)
        if not self.client.CLOUD_PLATFORM_SPACES and self.client.CPD_version < 4.8:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        if project_id:
            self.client.set.default_project(project_id)
        else:
            self.client.set.default_space(space_id)

        self.PromptTuningTypes = PromptTuningTypes
        self.PromptTuningInitMethods = PromptTuningInitMethods

        self.Tasks = (
            self._get_tasks_enum()
        )  # Note: Dynamically create enum with supported ENUM Tasks

        self.runs = TuneRuns(client=self.client)  # type: ignore[method-assign]

    def runs(self, *, filter: str) -> "TuneRuns":
        """Get historical tuning runs with the name filter.

        :param filter: filter, choose which runs specifying the tuning name to fetch
        :type filter: str

        **Examples**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(...)
            experiment.runs(filter='prompt tuning name').list()
        """
        return TuneRuns(client=self.client, filter=filter)

    def prompt_tuner(
        self,
        name: str,  # Note: Rest API does not require name,
        task_id: str,
        description: str | None = None,
        base_model: str | None = None,
        accumulate_steps: int | None = None,
        batch_size: int | None = None,
        init_method: str | None = None,
        init_text: str | None = None,
        learning_rate: float | None = None,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
        num_epochs: int | None = None,
        verbalizer: str | None = None,
        tuning_type: str | None = None,
        auto_update_model: bool = True,
        group_by_name: bool = False,
    ) -> PromptTuner:
        """Initialize a PromptTuner module.

        :param name: name for the PromptTuner
        :type name: str

        :param task_id: task that is targeted for this model. Example: `experiment.Tasks.CLASSIFICATION`

            Possible values:

            - experiment.Tasks.CLASSIFICATION: 'classification' (default)
            - experiment.Tasks.QUESTION_ANSWERING: 'question_answering'
            - experiment.Tasks.SUMMARIZATION: 'summarization'
            - experiment.Tasks.RETRIEVAL_AUGMENTED_GENERATION: 'retrieval_augmented_generation'
            - experiment.Tasks.GENERATION: 'generation'
            - experiment.Tasks.CODE_GENERATION_AND_CONVERSION: 'code'
            - experiment.Tasks.EXTRACTION: 'extraction

        :type task_id: str

        :param description: description
        :type description: str, optional

        :param base_model: model ID of the base model for this prompt tuning. Example: google/flan-t5-xl
        :type base_model: str, optional

        :param accumulate_steps: Number of steps to be used for gradient accumulation. Gradient accumulation
            refers to the method of collecting gradient for a configured number of steps instead of updating
            the model variables at every step and then applying the update to model variables.
            This can be used as a tool to overcome smaller batch size limitation.
            Often also referred in conjunction with "effective batch size". Possible values: 1 ≤ value ≤ 128,
            default value: 16
        :type accumulate_steps: int, optional

        :param batch_size: The batch size is the number of samples processed before the model is updated.
            Possible values: 1 ≤ value ≤ 16, default value: 16
        :type batch_size: int, optional

        :param init_method: `text` method requires `init_text` to be set. Allowable values: [`random`, `text`],
            default value: `random`
        :type init_method: str, optional

        :param init_text: initialization text to be used if `init_method` is set to `text`, otherwise this will be ignored.
        :type init_text: str, optional

        :param learning_rate: learning rate to be used while tuning prompt vectors. Possible values: 0.01 ≤ value ≤ 0.5,
            default value: 0.3
        :type learning_rate: float, optional

        :param max_input_tokens: maximum length of input tokens being considered. Possible values: 1 ≤ value ≤ 256,
            default value: 256
        :type max_input_tokens: int, optional

        :param max_output_tokens: maximum length of output tokens being predicted. Possible values: 1 ≤ value ≤ 128
            default value: 128
        :type max_output_tokens: int, optional

        :param num_epochs: number of epochs to tune the prompt vectors, this affects the quality of the trained model.
            Possible values: 1 ≤ value ≤ 50, default value: 20
        :type num_epochs: int, optional

        :param verbalizer: verbalizer template to be used for formatting data at train and inference time.
            This template may use brackets to indicate where fields from the data model must be rendered.
            The default value is "{{input}}" which means use the raw text, default value: `Input: {{input}} Output:`
        :type verbalizer: str, optional

        :param tuning_type: type of Peft (Parameter-Efficient Fine-Tuning) config to build.
            Allowable values: [`experiment.PromptTuningTypes.PT`], default value: `experiment.PromptTuningTypes.PT`
        :type tuning_type: str, optional

        :param auto_update_model: define if model should be automatically updated, default value: `True`
        :type auto_update_model: bool, optional

        :param group_by_name: define if tunings should be grouped by name, default value: `False`
        :type group_by_name: bool, optional

        **Examples**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(...)

            prompt_tuner = experiment.prompt_tuner(
                name="prompt tuning name",
                task_id=experiment.Tasks.CLASSIFICATION,
                base_model='google/flan-t5-xl',
                accumulate_steps=32,
                batch_size=16,
                learning_rate=0.2,
                max_input_tokens=256,
                max_output_tokens=2,
                num_epochs=6,
                tuning_type=experiment.PromptTuningTypes.PT,
                verbalizer="Extract the satisfaction from the comment. Return simple '1' for satisfied customer or '0' for unsatisfied. Input: {{input}} Output: ",
                auto_update_model=True)
        """

        if isinstance(task_id, Enum):
            task_id = task_id.value

        if isinstance(base_model, Enum):
            base_model = base_model.value

        if self.client._use_fm_ga_api:
            model_specs = (
                self.client.foundation_models.get_model_specs_with_prompt_tuning_support()
            )
        else:
            with catch_warnings():
                simplefilter("ignore", category=DeprecationWarning)
                model_specs = get_model_specs_with_prompt_tuning_support(
                    self.client.credentials.url  # type: ignore[arg-type]
                )

        prompt_tuning_supported_models = [
            model_spec["model_id"] for model_spec in model_specs.get("resources", [])  # type: ignore[union-attr]
        ]

        if base_model not in prompt_tuning_supported_models:
            raise WMLClientError(
                f"Base model '{base_model}' is not supported. Supported models: {prompt_tuning_supported_models}"
            )

        # check if model is in constricted mode
        _check_model_state(self.client, base_model, model_specs=model_specs)  # type: ignore[arg-type]

        prompt_tuner = PromptTuner(
            name=name,
            task_id=task_id,
            description=description,
            base_model=base_model,
            accumulate_steps=accumulate_steps,
            batch_size=batch_size,
            init_method=init_method,
            init_text=init_text,
            learning_rate=learning_rate,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            num_epochs=num_epochs,
            tuning_type=tuning_type,
            verbalizer=verbalizer,
            auto_update_model=auto_update_model,
            group_by_name=group_by_name,
        )

        prompt_tuner._client = self.client

        return prompt_tuner

    def fine_tuner(
        self,
        name: str,
        task_id: str,
        base_model: str | None = None,
        description: str | None = None,
        num_epochs: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        max_seq_length: int | None = None,
        accumulate_steps: int | None = None,
        verbalizer: str | None = None,
        response_template: str | None = None,
        gpu: dict | None = None,
        auto_update_model: bool = True,
        group_by_name: bool = False,
        peft_parameters: dict | PeftParameters | None = None,
        gradient_checkpointing: bool | None = None,
    ) -> FineTuner:
        """Initialize a FineTuner module.

        :param name: name for the FineTuner
        :type name: str

        :param base_model: model id of the base model for this fine-tuning.
        :type base_model: str

        :param task_id: task that is targeted for this model.
        :type task_id: str

        :param description: description
        :type description: str, optional

        :param num_epochs: number of epochs to tune the fine vectors, this affects the quality of the trained model.
            Possible values: 1 ≤ value ≤ 50, default value: 20
        :type num_epochs: int, optional

        :param learning_rate: learning rate to be used while tuning prompt vectors. Possible values: 0.01 ≤ value ≤ 0.5,
            default value: 0.3
        :type learning_rate: float, optional

        :param batch_size: The batch size is a number of samples processed before the model is updated.
            Possible values: 1 ≤ value ≤ 16, default value: 16
        :type batch_size: int, optional

        :param max_seq_length:
        :type max_seq_length: int, optional

        :param accumulate_steps: Number of steps to be used for gradient accumulation. Gradient accumulation
            refers to a method of collecting gradient for configured number of steps instead of updating
            the model variables at every step and then applying the update to model variables.
            This can be used as a tool to overcome smaller batch size limitation.
            Often also referred in conjunction with "effective batch size". Possible values: 1 ≤ value ≤ 128,
            default value: 16
        :type accumulate_steps: int, optional

        :param verbalizer: verbalizer template to be used for formatting data at train and inference time.
        :type verbalizer: str, optional

        :param response_template: Separator for the prediction/response in the single sequence to train on completions only.
        :type response_template: str, optional

        :param gpu: The name and number of GPUs used for the FineTuning job.
        :type gpu: dict, optional

        :param auto_update_model: define if model should be automatically updated, default value: `True`
        :type auto_update_model: bool, optional

        :param group_by_name: define if tunings should be grouped by name, default value: `False`
        :type group_by_name: bool, optional

        :param peft_parameters: parameters to be set when running a Fine Tuning job with LoRA/QLoRA
        :type peft_parameters: dict, PeftParameters, optional

        :param gradient_checkpointing: reduces GPU memory required at the cost of slowing training by approx 20%
        :type gradient_checkpointing: bool, optional

        **Examples**

        Example of init FineTuner

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(...)

            fine_tuner = experiment.fine_tuner(
                name="fine-tuning name",
                base_model="bigscience/bloom-560m",
                task_id="generation",
                num_epochs=3,
                learning_rate=0.2,
                batch_size=5,
                max_seq_length=1024,
                accumulate_steps=5,
                verbalizer="### Input: {{input}} \\n\\n### Response: {{output}}",
                response_template="\\n### Response:\\n",
                auto_update_model=False,
                gradient_checkpointing=True)

        Example of init FineTuner with LoRA/QLoRA

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment
            from ibm_watsonx_ai.foundation_models.schema import PeftParameters

            experiment = TuneExperiment(...)

            fine_tuner = experiment.fine_tuner(
                name="fine-tuning name",
                base_model="meta-llama/Meta-Llama-3-8B",
                task_id="classification",
                num_epochs=3,
                learning_rate=0.2,
                batch_size=5,
                max_seq_length=1024,
                accumulate_steps=5,
                verbalizer="### Input: {{input}} \\n\\n### Response: {{output}}",
                response_template="\\n### Response:\\n",
                peft_parameters=PeftParameters(
                    type="lora",
                    rank=8,
                    lora_alpha= 2,
                    lora_dropout=0.05,
                    target_modules=['all-linear'],
                ),
                auto_update_model=False,
                gradient_checkpointing=False)
        """

        if isinstance(task_id, Enum):
            task_id = task_id.value

        if isinstance(base_model, Enum):
            base_model = base_model.value

        fine_tuner = FineTuner(
            name=name,
            description=description,
            base_model=base_model,
            task_id=task_id,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            accumulate_steps=accumulate_steps,
            verbalizer=verbalizer,
            response_template=response_template,
            gpu=gpu,
            auto_update_model=auto_update_model,
            group_by_name=group_by_name,
            api_client=self.client,
            peft_parameters=peft_parameters,
            gradient_checkpointing=gradient_checkpointing,
        )

        return fine_tuner

    def ilab_tuner(
        self,
        name: str,
    ) -> ILabTuner:
        """Initialize a ILabTuner module.

        :param name: name for the ILabTuner
        :type name: str

        **Examples**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(...)

            prompt_tuner = experiment.ilab_tuner(name="ilab-tuning name")
        """

        ilab_tuner = ILabTuner(name=name, api_client=self.client)

        return ilab_tuner

    def _get_tasks_enum(self) -> EnumLike:
        try:
            from ibm_watsonx_ai.foundation_models.utils.utils import (
                get_all_supported_tasks_dict,
            )

            return Enum(
                value="TuneExperimentTasks",
                names=get_all_supported_tasks_dict(url=self.client.credentials.url),
            )
        except Exception:
            return TuneExperimentTasks
