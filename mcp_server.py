import os
import json
import logging
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ── MCP Server Setup ──────────────────────────────────────────────────────────
mcp = FastMCP("kids-content-server")


# ── Tool 1: Save Locally ──────────────────────────────────────────────────────
@mcp.tool()
def save_local(folder: str, filename: str, content: dict) -> str:
    """Save content as JSON to local stories/ folder."""
    os.makedirs(folder, exist_ok=True)
    path = f"{folder}/{filename}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)
    logger.info(f"[MCP] Saved locally: {path}")
    return f"Saved to {path}"


# ── Tool 2: Upload to Google Drive ────────────────────────────────────────────
@mcp.tool()
def upload_to_drive(file_path: str, drive_folder_name: str = "Kids Stories") -> str:
    """Upload a file to Google Drive folder."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    import pickle

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]
    creds = None

    # Load existing token if available
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    # If no valid credentials, do OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            # creds = flow.run_local_server(port=0)
            creds = flow.run_local_server(port=8080, open_browser=True)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    service = build("drive", "v3", credentials=creds)

    # Find or create the folder in Drive
    query = f"name='{drive_folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get("files", [])

    if folders:
        folder_id = folders[0]["id"]
    else:
        folder_metadata = {
            "name": drive_folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        folder = service.files().create(body=folder_metadata, fields="id").execute()
        folder_id = folder["id"]

    # Upload the file
    file_name = os.path.basename(file_path)
    media = MediaFileUpload(file_path, resumable=True)
    file_metadata = {"name": file_name, "parents": [folder_id]}
    uploaded = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id, name, webViewLink"
    ).execute()

    link = uploaded.get("webViewLink", "")
    logger.info(f"[MCP] Uploaded to Drive: {file_name} → {link}")
    return f"Uploaded {file_name} to Drive folder '{drive_folder_name}': {link}"


# ── Run Server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="stdio")