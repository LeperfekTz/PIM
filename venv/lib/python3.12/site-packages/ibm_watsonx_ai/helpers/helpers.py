#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import json
from configparser import ConfigParser
from typing import Union

__all__ = [
    "get_credentials_from_config",
    "pipeline_to_script",
]


def get_credentials_from_config(env_name, credentials_name, config_path="./config.ini"):
    """Load credentials from the config file.

    ::

        [DEV_LC]

        credentials = { }
        cos_credentials = { }

    :param env_name: name of [ENV] defined in the config file
    :type env_name: str
    :param credentials_name: name of credentials
    :type credentials_name: str
    :param config_path: path to the config file
    :type config_path: str
    :return: loaded credentials
    :rtype: dict

    **Example:**

    .. code-block:: python

        get_credentials_from_config(env_name='DEV_LC', credentials_name='credentials')

    """
    config = ConfigParser()
    config.read(config_path)

    return json.loads(config.get(env_name, credentials_name))


def pipeline_to_script(pipeline) -> Union["str", "HTML"]:
    """Create a python script based on a passed pipeline model. (Python code representation of pipeline model)

    :param pipeline: pipeline model to be written as a script
    :type pipeline: Pipeline or TrainedPipeline

    :return: information about the script location
    :rtype: str or html

    **Example:**

    .. code-block:: python

        pipeline_to_script(pipeline=best_pipeline)
    """
    from lale.helpers import import_from_sklearn_pipeline
    from sklearn.pipeline import Pipeline
    from ibm_watsonx_ai.utils.autoai.utils import is_ipython
    from ibm_watsonx_ai.utils import create_download_link
    import os

    script_name = "pipeline_script.py"

    if isinstance(pipeline, Pipeline):
        pipeline = import_from_sklearn_pipeline(pipeline)

    script = pipeline.pretty_print()

    with open(script_name, "w") as f:
        f.write(script)

    script_location = f"{os.path.abspath('.')}/{script_name}"

    if is_ipython():
        return create_download_link(script_location)
    else:
        return f"Pipeline python script location: {script_location}"
