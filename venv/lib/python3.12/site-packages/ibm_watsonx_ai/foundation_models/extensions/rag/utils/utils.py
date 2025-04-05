#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import sys
import ssl
import ast
import base64
import pandas as pd
import logging
import tempfile
from typing import TYPE_CHECKING, cast, Any, Callable
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai.wml_client_error import MissingValue, UnexpectedKeyWordArgument
from ibm_watsonx_ai.foundation_models.schema import BaseSchema
from ibm_watsonx_ai.utils.utils import _get_default_args

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from ibm_watsonx_ai.foundation_models import ModelInference

logger = logging.getLogger(__name__)


# Verbose display in notebooks


def verbose_search(
    question: str, documents: list[Document] | list[tuple[Document, float]]
) -> None:
    """Display a table with found documents.

    :param question: question/query used for search
    :type question: str
    :param documents: list of documents found with question or list of tuples (if search was done with scores)
    :type documents: list[langchain_core.documents.Document] | list[Document, float]
    :raises ImportError: if it is notebook environment but IPython is not found
    """

    # Unzip if list have tuples (Document, float)
    if all(isinstance(doc, tuple) for doc in documents):
        documents, scores = [doc[0] for doc in documents], [doc[1] for doc in documents]  # type: ignore[index]
    else:
        scores = []

    from langchain_core.documents import Document

    documents = cast(list[Document], documents)
    if "ipykernel" in sys.modules:
        try:
            from IPython.display import display, Markdown
        except ImportError:
            raise ImportError(
                "To use verbose search, please install make sure IPython package is installed."
            )

        display(Markdown(f"**Question:** {question}"))

        if len(documents) > 0:
            metadata_fields: set = set()

            for doc in documents:
                metadata_fields.update(doc.metadata.keys())

            if scores:
                metadata_fields.add("score")

            df = pd.DataFrame(columns=["page_content"] + list(metadata_fields))

            # Parsing rows and adding them to the DataFrame
            for doc in documents:
                row = {"page_content": doc.page_content}
                row.update(doc.metadata)
                # Adding score (if provided)
                if scores:
                    row["score"] = scores.pop(0)
                df = pd.concat(
                    [df, pd.DataFrame({key: [value] for key, value in row.items()})],
                    ignore_index=True,
                )

            display(df)
        else:
            display(Markdown("No documents were found."))
    else:
        if len(documents) > 0:
            if scores:
                for i, (d, s) in enumerate(zip(documents, scores)):
                    logger.info(f"{i} | {s} |  {d.page_content}   | {d.metadata}")
            else:
                for i, d in enumerate(documents):
                    logger.info(f"{i} |  {d.page_content}   | {d.metadata}")
        else:
            logger.info("No documents were found.")


# SSL Certificates


def is_valid_certificate(cert_string: str) -> bool:
    try:
        ssl.PEM_cert_to_DER_cert(cert_string)
        return True
    except Exception:
        return False


def get_ssl_certificate(cert: str) -> str:
    if is_valid_certificate(cert):
        return cert
    else:
        try:
            cert_padded = cert + "==="
            cert_decoded = base64.b64decode(cert_padded).decode()
            if is_valid_certificate(cert_decoded):
                return cert_decoded
            else:
                raise ValueError("Not a valid SSL certificate.")
        except Exception as e:
            raise ValueError(
                f"Error occured when trying to get the SSL certificate: {e}"
            )


def save_ssl_certificate_as_file(ssl_certificate_content: str) -> str:
    ssl_certificate_content = get_ssl_certificate(ssl_certificate_content)
    with tempfile.NamedTemporaryFile(suffix=".crt", delete=False) as file:
        file.write(ssl_certificate_content.encode())

    logger.info(
        f"SSL certificate was found and written to {file.name}. It will be used for the connection for the VectorStore."
    )
    return file.name


def get_max_input_tokens(
    model: ModelInference, params: dict | None = None, **kwargs: Any
) -> int:
    """
    Get maximum number of tokens allowed as input for a given model.

    :param model: Initialized :class:`ModelInference <ibm_watsonx_ai.foundation_models.inference.model_inference.ModelInference>` object.
    :type model: ModelInference

    :param params: A dictionary containing the request parameters.
    :type params: dict

    :return: The maximum number of tokens allowed as input.
    :rtype: int

    :raises MissingValue: If the model limits cannot be found in the model details.
    """
    supported_kwargs = ["default_max_sequence_length"]
    if unsupported_kwargs := set(kwargs.keys()) - set(supported_kwargs):
        for kword in unsupported_kwargs:
            raise UnexpectedKeyWordArgument(
                kword,
                reason=f"{kword} is not supported as a keyword argument. Supported kwargs: {supported_kwargs}",
            )

    model_max_sequence_length = (
        model.get_details().get("model_limits", {}).get("max_sequence_length")
    )

    warn_msg = f"Model `{model.model_id}` limits cannot be found in the model details"

    if model_max_sequence_length is None:
        logger.warning(warn_msg)
        # use default param if passed
        model_max_sequence_length = (params or kwargs).get(
            "default_max_sequence_length"
        )

    if model_max_sequence_length is None:
        raise MissingValue(
            value_name="model_limits",
            reason=warn_msg,
        )

    if isinstance(model.params, BaseSchema):
        params_dict = model.params.to_dict()
    else:
        params_dict = model.params or {}

    model_max_new_tokens = params_dict.get(GenTextParamsMetaNames.MAX_NEW_TOKENS, 20)
    return model_max_sequence_length - model_max_new_tokens


class FunctionTransformer(ast.NodeTransformer):
    """Class for injecting data into function

    :param function: function to be transformed
    :type function: ast.FunctionDef
    """

    def __init__(self, function: ast.FunctionDef, **kwargs: Any) -> None:
        self.args_to_replace = kwargs
        self.function_args = [arg.arg for arg in function.args.args]

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Visits node of type Call and injects values if needed

        :param node: Call node
        :type node: ast.Call

        :return: the same node after injecting missing values or left as is
        :rtype: ast.AST
        """
        new_keywords: list[ast.keyword] = []
        for keyword in node.keywords:
            new_keyword = None
            if (
                hasattr(keyword.value, "id")
                and keyword.value.id in self.args_to_replace
            ):
                v: dict = self.args_to_replace.get(keyword.value.id, {})
                replace = v.get("replace")
                value = v.get("value")
                if replace:
                    if isinstance(value, dict):
                        new_keyword = ast.keyword(
                            arg=keyword.arg,
                            value=ast.parse(str(value), filename="tmp", mode="exec")
                            .body[0]
                            .value,  # type: ignore[attr-defined]
                        )
                    else:
                        new_keyword = ast.keyword(
                            arg=keyword.arg, value=ast.Constant(value=value)
                        )
            else:
                new_keyword = keyword

            if new_keyword and new_keyword.arg not in [el.arg for el in new_keywords]:
                new_keywords.append(new_keyword)

        node.keywords = new_keywords
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST | ast.Assign:
        """Visits node of type Assign and injects values if needed

        :param node: object representing the Assign node
        :type node: ast.Assign

        :return: the same node after injecting missing values or left as is
        :rtype: ast.AST | ast.Assign
        """
        for target in node.targets:
            if hasattr(node.value, "id") and node.value.id in self.args_to_replace:
                if node.value.id.endswith("input_data_references".upper()):
                    return ast.Assign(
                        targets=[target],
                        value=ast.List(
                            [
                                ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(
                                            id="DataConnection", ctx=ast.Load()
                                        ),
                                        attr="from_dict",
                                        ctx=ast.Load(),
                                    ),
                                    args=[ast.parse(str(el), filename="tmp", mode="exec").body[0].value],  # type: ignore[attr-defined]
                                    keywords=[],
                                )
                                for el in self.args_to_replace[node.value.id]
                            ],
                            ctx=ast.Load(),
                        ),
                        lineno=target.lineno,
                    )
                else:
                    new_value = self.args_to_replace[node.value.id]
                    if isinstance(new_value, dict):
                        value = (
                            ast.parse(str(new_value), filename="tmp", mode="exec")
                            .body[0]
                            .value  # type: ignore[attr-defined]
                        )
                    else:
                        value = ast.Constant(value=new_value)
                    return ast.Assign(
                        targets=[target],
                        value=value,
                        lineno=target.lineno,
                    )
        return self.generic_visit(node)


class FunctionVisitor(ast.NodeVisitor):
    """Class for mapping function definition into nodes"""

    def __init__(self) -> None:
        self.function: ast.FunctionDef | None = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Helper fo visits node of type functionDef.

        :param node: FunctionDef node
        :type node: ast.FunctionDef
        """
        self.function = node
        self.generic_visit(node)


def _get_components_replace_data(
    init_params: dict, func: Callable, suffix: str
) -> dict:
    """Helper for selecting minimum set of parameters from `init_params` object
    that need to be passed to func to assure proper and unique call.

    :param init_params: the set of parameters to be passed to the func object.
    :type init_params: dict

    :param func: function object whose parameters are analyzed
    :type func: Callable

    :param suffix: suffix added after `REPLACE_THIS_CODE_WITH_` to distinguish between parameter groups to be replaced in template
    :type suffix: str

    :return: a set of func parameters with information whether they are to be passed
    and under what name they can be located in the template. Sample record
        {
            f"REPLACE_THIS_CODE_WITH_{suffix.upper()}_{arg.upper()}": {
                "value": init_params.get(arg),
                "replace": False,
            }
        }
    :rtype: dict
    """
    new_model_init_params = {}
    init_params_defaults = _get_default_args(func)
    for arg, value in init_params_defaults.copy().items():
        if arg in init_params and value == init_params[arg]:
            new_model_init_params.update(
                {
                    f"REPLACE_THIS_CODE_WITH_{suffix.upper()}_{arg.upper()}": {
                        "value": init_params.get(arg),
                        "replace": False,
                    }
                }
            )
        elif arg in init_params:
            new_model_init_params.update(
                {
                    f"REPLACE_THIS_CODE_WITH_{suffix.upper()}_{arg.upper()}": {
                        "value": init_params.get(arg),
                        "replace": True,
                    }
                }
            )
        else:
            new_model_init_params.update(
                {
                    f"REPLACE_THIS_CODE_WITH_{suffix.upper()}_{arg.upper()}": {
                        "value": init_params.get(arg),
                        "replace": False,
                    }
                }
            )

    return new_model_init_params
