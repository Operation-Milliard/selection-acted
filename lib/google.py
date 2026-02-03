from __future__ import annotations

import base64
import os
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from io import BytesIO
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]


def get_credentials(oauth_client_path: str, token_path: str):
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(oauth_client_path, SCOPES)
            creds = flow.run_local_server(port=0)
        os.makedirs(os.path.dirname(token_path) or ".", exist_ok=True)
        with open(token_path, "w", encoding="utf-8") as handle:
            handle.write(creds.to_json())

    return creds


def get_sheet_values(credentials, spreadsheet_id: str, range_a1: str) -> list[list[str]]:
    service = build("sheets", "v4", credentials=credentials)
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=range_a1)
        .execute()
    )
    return result.get("values", [])


def update_sheet_values(credentials, spreadsheet_id: str, range_a1: str, values: list[list[str]]):
    service = build("sheets", "v4", credentials=credentials)
    body = {"values": values}
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=range_a1,
        valueInputOption="RAW",
        body=body,
    ).execute()


def list_files_in_folder(credentials, folder_id: str) -> list[dict]:
    service = build("drive", "v3", credentials=credentials)
    query = f"'{folder_id}' in parents and trashed = false"
    files = []
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageToken=page_token,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return files


def download_drive_file(credentials, file_id: str) -> bytes:
    service = build("drive", "v3", credentials=credentials)
    request = service.files().get_media(fileId=file_id)
    buffer = BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def list_sheet_titles(credentials, spreadsheet_id: str) -> set[str]:
    service = build("sheets", "v4", credentials=credentials)
    response = service.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        fields="sheets(properties(title))",
    ).execute()
    return {sheet["properties"]["title"] for sheet in response.get("sheets", [])}


def clear_sheet(credentials, spreadsheet_id: str, title: str) -> None:
    service = build("sheets", "v4", credentials=credentials)
    service.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=title,
        body={},
    ).execute()


def add_sheet(credentials, spreadsheet_id: str, title: str) -> None:
    service = build("sheets", "v4", credentials=credentials)
    body = {"requests": [{"addSheet": {"properties": {"title": title}}}]}
    service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()


def send_email_with_attachment(
    credentials,
    to: str,
    subject: str,
    body: str,
    attachment_path: Path,
    from_name: str | None = None,
) -> dict:
    """Send email via Gmail API with HTML attachment."""
    message = MIMEMultipart()
    message["to"] = to
    message["subject"] = subject
    if from_name:
        message["from"] = from_name

    message.attach(MIMEText(body, "plain", "utf-8"))

    with open(attachment_path, "rb") as f:
        part = MIMEBase("text", "html")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f"attachment; filename={attachment_path.name}",
    )
    message.attach(part)

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    service = build("gmail", "v1", credentials=credentials)
    return service.users().messages().send(userId="me", body={"raw": raw}).execute()
