import logging
import mimetypes
import os
from datetime import datetime, timedelta
from io import BytesIO
from time import sleep
from urllib.parse import urljoin

import magic
import requests
from django.conf import settings
from django.core.cache import caches
from django.core.exceptions import ImproperlyConfigured
from django.core.files.base import ContentFile
from django.utils.translation import gettext as _
from PIL import Image, UnidentifiedImageError
from requests.exceptions import ChunkedEncodingError

from geotrek.common.parsers import (
    AttachmentParserMixin,
    DownloadImportError,
    Parser
)

logger = logging.getLogger(__name__)


class ZohoParser(Parser):
    base_auth_url = "https://accounts.zoho.com/oauth/v2/token"
    access_token_error_status = [401]
    access_token_cache = caches["default"]
    access_token_cache_key = "zoho_crm_access_token"

    # Configuration required in custom parser, for OAuth 2.0:
    client_id = None
    client_secret = None
    soid = None

    # Configuration required in custom parser, for building urls:
    url = None  # Zoho CRM 'api_domain'
    lookup_module = None  # String - module api name (ex: "Adh_sions")

    headers = {}

    def __init__(self, *args, **kwargs):
        self.check_auth_config()
        self.set_access_token_header()
        self.fields_param = self.get_fields_param()
        self.nb = self.get_record_count()
        super().__init__(*args, **kwargs)

    def check_auth_config(self):
        required_attributes = ["client_id", "client_secret", "soid"]
        missing_attributes = [
            attr for attr in required_attributes if getattr(self, attr) is None
        ]
        if missing_attributes:
            msg = _(
                "The following attributes are missing from parser configuration: %(attributes)s"
            ) % {"attributes": ", ".join(missing_attributes)}
            raise ImproperlyConfigured(msg)

    def set_access_token_header(self, force_generation=False):
        cached_value = self.access_token_cache.get(self.access_token_cache_key)
        if (
            force_generation
            or cached_value is None
            or cached_value["creation_date"] + timedelta(minutes=59) < datetime.now()
        ):
            access_token = self.generate_access_token()
        else:
            access_token = cached_value["token"]
        self.headers["Authorization"] = f"Zoho-oauthtoken {access_token}"

    def generate_access_token(self):
        """
        Handles Zoho OAuth 2.0 using their "Self Client - Client Credentials flow":
        https://www.zoho.com/accounts/protocol/oauth/self-client/client-credentials-flow.html
        """
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
            "scope": "ZohoCRM.modules.READ",
            "soid": self.soid,
        }
        response = self.request_or_retry(self.base_auth_url, "post", data=payload)
        json_response = response.json()
        if json_response.get("error") is not None:
            msg = _(
                "Could not generate an access token. Please check your parser's OAuth 2.0 configuration. Error: %(error)s"
            ) % {"error": json_response["error"]}
            raise ImproperlyConfigured(msg)
        access_token = json_response["access_token"]
        self.access_token_cache.set(
            self.access_token_cache_key,
            {"token": access_token, "creation_date": datetime.now()},
        )
        return access_token

    def handle_access_token_error(self, response):
        logger.info(
            "Failed to fetch %s. Invalid or expired token. Regenerating token and retrying...",
            response.url,
        )
        self.set_access_token_header(force_generation=True)

    def get_fields_param(self):
        """
        Generate a string corresponding to the 'fields' parameter for fetching records.
        One ZohoCRM request can include a maximum of 50 field API names.
        """
        # Get list of the api fields used in this parser:
        field_list = {**self.fields, **self.non_fields}.values()
        # Flatten the list (some values can be tuples):
        flattened_field_list = []
        for field in field_list:
            if isinstance(field, tuple):
                flattened_field_list.extend(field)
            else:
                flattened_field_list.append(field)
        # Remove duplicates:
        unique_field_set = set(flattened_field_list)

        if len(unique_field_set) > 50:
            # TODO: if more than 50 unique fields, make several requests and
            # zip the results by id?
            msg = _(
                "ZohoParser supports a maximum of 50 unique API fields. Your configuration exceeds this limit, so records cannot be fetched."
            )
            raise NotImplementedError(msg)

        # The fields parameter should end with a comma:
        return ",".join(unique_field_set) + ","

    def request_or_retry(self, url, verb="get", **kwargs):
        """This `Parser` method is overriden so the access token can be refreshed when it expires"""
        
        def prepare_retry(error_msg):
            logger.info("Failed to fetch %s. %s. Retrying...", url, error_msg)
            sleep(settings.PARSER_RETRY_SLEEP_TIME)

        def format_exception(e):
            return _("%(error_name)s: %(error_msg)s") % {
                "error_name": e.__class__.__name__,
                "error_msg": e,
            }

        action = getattr(requests, verb)
        response = None
        try_get = 0

        while try_get < settings.PARSER_NUMBER_OF_TRIES:
            try_get += 1
            try:
                response = action(
                    url, headers=self.headers, allow_redirects=True, **kwargs
                )
                if response.status_code == 200:
                    return response
                elif response.status_code in settings.PARSER_RETRY_HTTP_STATUS:
                    prepare_retry(f"Status code: {response.status_code}")
                elif response.status_code in self.access_token_error_status:
                    self.handle_access_token_error(response)
                else:
                    logger.info(
                        "Failed to fetch %s. Status code: %s.",
                        url,
                        response.status_code,
                    )
                    break
            except (ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
                prepare_retry(format_exception(e))
            except Exception as e:
                raise DownloadImportError(
                    _("Failed to fetch %(url)s. %(error)s")
                    % {
                        "url": url,
                        "error": format_exception(e),
                    }
                )
        logger.warning("Failed to fetch %s after %s attempt(s).", url, try_get)
        msg = _("Failed to fetch %(url)s after %(try_get)s attempt(s).") % {
            "url": url,
            "try_get": try_get,
        }
        raise DownloadImportError(msg)

    def get_record_count(self):
        """https://www.zoho.com/crm/developer/docs/api/v8/module_record_count.html"""
        url = urljoin(self.url, f"crm/v8/{self.lookup_module}/actions/count")
        response = self.request_or_retry(url)
        return response.json()["count"]

    def next_row(self):
        def fetch_records():
            """https://www.zoho.com/crm/developer/docs/api/v8/get-records.html"""
            page_token_param = (
                f"&page_token={self.next_page_token}" if self.next_page_token else ""
            )
            url = urljoin(
                self.url,
                f"crm/v8/{self.lookup_module}?fields={self.fields_param}{page_token_param}",
            )
            response = self.request_or_retry(url)
            self.root = response.json()
            self.next_page_token = self.root["info"]["next_page_token"]
            return self.root["data"]

        self.next_page_token = None
        yield from fetch_records()
        while self.next_page_token:
            yield from fetch_records()

    def normalize_field_name(self, name):
        return name


class ZohoAttachmentParserMixin(AttachmentParserMixin):
    """
    https://www.zoho.com/crm/developer/docs/api/v8/download_field_attachments.html
    """

    def get_image_download_url(self, image_id):
        if image_id is None:
            return None
        return urljoin(
            self.url,
            f"crm/v8/{self.lookup_module}/{self.obj.eid}/actions/download_fields_attachment?fields_attachment_id={image_id}",
        )

    def check_attachment_updated(self, attachments_to_delete, updated, **kwargs):
        """
        Ugly workaround since file size cannot be obtained without downloading the
        file due to the chunked transfer encoding.
        """
        return False, updated

    def generate_content_attachment(self, attachment, parsed_url, url, updated, name):
        """
        This `AttachmentParserMixin` method is overriden because the filename is not
        present in the download url. We assume that `attachment.title` corresponds
        to the filename and contains the extension.
        """
        if (
            parsed_url.scheme in ("http", "https") and self.download_attachments
        ) or parsed_url.scheme == "ftp":
            content = self.download_attachment(url)
            if content is None:
                return False, updated
            f = ContentFile(content)
            if (
                settings.PAPERCLIP_MAX_BYTES_SIZE_IMAGE
                and settings.PAPERCLIP_MAX_BYTES_SIZE_IMAGE < f.size
            ):
                self.add_warning(
                    _("%(class)s #%(pk)s - %(url)s: downloaded file is too large")
                    % {
                        "url": url,
                        "pk": self.obj.pk,
                        "class": self.obj.__class__.__name__,
                    }
                )
                return False, updated
            try:
                image = Image.open(BytesIO(content))
                if (
                    settings.PAPERCLIP_MIN_IMAGE_UPLOAD_WIDTH
                    and settings.PAPERCLIP_MIN_IMAGE_UPLOAD_WIDTH > image.width
                ):
                    self.add_warning(
                        _(
                            "%(class)s #%(pk)s - {url}: downloaded file is not wide enough"
                        )
                        % {
                            "url": url,
                            "pk": self.obj.pk,
                            "class": self.obj.__class__.__name__,
                        }
                    )
                    return False, updated
                if (
                    settings.PAPERCLIP_MIN_IMAGE_UPLOAD_HEIGHT
                    and settings.PAPERCLIP_MIN_IMAGE_UPLOAD_HEIGHT > image.height
                ):
                    self.add_warning(
                        _(
                            "%(class)s #%(pk)s - %(url)s : downloaded file is not tall enough"
                        )
                        % {
                            "url": url,
                            "pk": self.obj.pk,
                            "class": self.obj.__class__.__name__,
                        }
                    )
                    return False, updated
                if settings.PAPERCLIP_ALLOWED_EXTENSIONS is not None:
                    filename = attachment.title
                    extension = os.path.splitext(filename)[1] if filename else None
                    extension = extension.lower().strip(".") if extension else None
                    if extension not in settings.PAPERCLIP_ALLOWED_EXTENSIONS:
                        self.add_warning(
                            _(
                                "Invalid attachment file %(url)s for %(class)s #%(pk)s: "
                                "File type '%(ext)s' is not allowed."
                            )
                            % {
                                "url": url,
                                "ext": extension,
                                "pk": self.obj.pk,
                                "class": self.obj.__class__.__name__,
                            }
                        )
                        return False, updated
                    f.seek(0)
                    file_mimetype = magic.from_buffer(f.read(), mime=True)
                    file_mimetype_allowed = (
                        f".{extension}" in mimetypes.guess_all_extensions(file_mimetype)
                    )
                    file_mimetype_allowed = file_mimetype_allowed or (
                        settings.PAPERCLIP_EXTRA_ALLOWED_MIMETYPES.get(extension, False)
                        and file_mimetype
                        in settings.PAPERCLIP_EXTRA_ALLOWED_MIMETYPES.get(extension)
                    )
                    if not file_mimetype_allowed:
                        self.add_warning(
                            _(
                                "Invalid attachment file %(url)s for %(class)s #%(pk)s: "
                                "File mime type '%(mimetype)s' is not allowed for %(ext)s."
                            )
                            % {
                                "url": url,
                                "ext": extension,
                                "mimetype": file_mimetype,
                                "pk": self.obj.pk,
                                "class": self.obj.__class__.__name__,
                            }
                        )
                        return False, updated
            except UnidentifiedImageError:
                pass
            except ValueError:
                # We want to catch : https://github.com/python-pillow/Pillow/blob/22ef8df59abf461824e4672bba8c47137730ef57/src/PIL/PngImagePlugin.py#L143
                return False, updated
            attachment.attachment_file.save(name, f, save=False)
            attachment.is_image = attachment.is_an_image()
        else:
            attachment.attachment_link = url
        return True, updated
