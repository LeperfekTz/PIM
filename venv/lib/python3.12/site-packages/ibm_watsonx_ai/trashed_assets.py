#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from pandas import DataFrame

from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class TrashedAssets(WMLResource):
    """Manage trashed assets."""

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)

    def get_details(
        self,
        asset_id: str | None = None,
        limit: int | None = None,
    ) -> dict:
        """Get metadata of a given trashed asset. If no `asset_id` is specified, all trashed assets metadata is returned.

        :param asset_id: trashed asset ID
        :type asset_id: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: export metadata
        :rtype: dict (if asset_id is not None) or {"resources": [dict]} (if asset_id is None)

        **Example:**

        .. code-block:: python

            details = client.trashed_assets.get_details(asset_id)
            details = client.trashed_assets.get_details()
            details = client.trashed_assets.get_details(limit=100)

        """

        TrashedAssets._validate_type(asset_id, "asset_id", str, False)
        TrashedAssets._validate_type(limit, "limit", int, False)

        href = self._client.service_instance._href_definitions.get_trashed_assets_href()

        if asset_id is None:
            return self._get_artifact_details(
                href,
                asset_id,
                limit,
                "trashed assets",
                query_params=self._client._params(),
                _async=False,
                _all=False,
            )

        else:
            return self._get_artifact_details(
                href,
                asset_id,
                limit,
                "trashed assets",
                query_params=self._client._params(),
            )

    @staticmethod
    def _prepare_attachment_list_for_removal(asset_details):
        def attachment_requires_removal(attachment_details):
            return attachment_details.get("is_remote") or (
                attachment_details.get("is_referenced")
                and attachment_details.get("is_object_key_read_only")
            )

        return list(
            filter(attachment_requires_removal, asset_details.get("attachments", []))
        )

    def _remove_attachment(self, attachment_path):
        response = self._client._session.delete(
            self._client.service_instance._href_definitions.get_wsd_model_attachment_href()
            + f"/{attachment_path}",
            headers=self._client._get_headers(),
            params=self._client._params(),
        )

        self._handle_response(204, "deleting attachment", response, json_response=False)

        return "SUCCESS"

    def list(self, limit: int | None = None) -> DataFrame:
        """List trashed assets.

        :param limit: set the limit for number of listed trashed assets,
            default is `None` (all trashed assets should be fetched)
        :type limit: int, optional

        :return: Pandas DataFrame with information about trashed assets
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            trashed_assets_list = client.trashed_assets.list()
            print(trashed_assets_list)

            # Result:
            #        NAME  ASSET_TYPE                              ASSET_ID
            # 0  data.csv  data_asset  8e421c27-767d-4824-9aab-dc5c7c19ba87

        """
        trashed_assets_details = self.get_details(limit=limit)
        trashed_assets_values = [
            (
                asset["metadata"]["name"],
                asset["metadata"]["asset_type"],
                asset["metadata"]["asset_id"],
            )
            for asset in trashed_assets_details["resources"]
        ]

        table = self._list(
            trashed_assets_values, ["NAME", "ASSET_TYPE", "ASSET_ID"], None
        )
        return table

    def restore(self, asset_id: str) -> dict:
        """Restore a trashed asset.

        :param asset_id: trashed asset ID
        :type asset_id: str

        :return: details of restored asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            asset_details = client.trashed_assets.restore(asset_id)
        """
        response = self._client._session.post(
            self._client.service_instance._href_definitions.get_trashed_asset_restore_href(
                asset_id
            ),
            headers=self._client._get_headers(),
            params=self._client._params(),
        )

        return self._handle_response(200, "restoring trashed asset", response)

    def purge_all(self) -> Literal["SUCCESS"]:
        """Purge all trashed asset.

        .. note::
            If there is more than 20 trashed assets, they will be removed asynchronously.
            It may take a few seconds until all trashed assets will disappear from trashed assets list.

        :return: status "SUCCESS" if purge is successful
        :rtype: Literal["SUCCESS"]

        **Example:**

        .. code-block:: python

            client.trashed_assets.purge_all()
        """
        response = self._client._session.delete(
            self._client.service_instance._href_definitions.get_trashed_assets_purge_all_href(),
            headers=self._client._get_headers(),
            params=self._client._params(),
        )

        if response.status_code in (202, 204):
            return "SUCCESS"
        else:
            self._handle_response(
                204, "purging all trashed assets", response, json_response=False
            )

    def delete(self, asset_id: str) -> Literal["SUCCESS"]:
        """Delete a trashed asset.

        :param asset_id: trashed asset ID
        :type asset_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]

        **Example:**

        .. code-block:: python

            client.trashed_assets.delete(asset_id)
        """
        response = self._client._session.delete(
            self._client.service_instance._href_definitions.get_trashed_asset_href(
                asset_id
            ),
            headers=self._client._get_headers(),
            params=self._client._params(),
        )

        self._handle_response(
            204, "deleting trashed asset", response, json_response=False
        )

        return "SUCCESS"

    @staticmethod
    def get_id(trashed_asset_details: dict) -> str:
        """Get the ID of a trashed asset.

        :param trashed_asset_details: metadata of the trashed asset
        :type trashed_asset_details: dict

        :return: unique ID of the trashed asset
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_id = client.trashed_assets.get_id(trashed_asset_details)

        """
        return trashed_asset_details["metadata"]["asset_id"]
