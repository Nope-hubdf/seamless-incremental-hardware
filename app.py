import hmac
import hashlib
import json
import os
import requests
from fastapi import FastAPI, Request, HTTPException, Response, Query

# --- Configuration ---
# Your shared secret.
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

# The URL of the final destination where the webhook should be sent.
# This will be used as a fallback if no query parameter is provided.
WEBHOOK_DESTINATION = os.getenv("WEBHOOK_DESTINATION")

app = FastAPI()

def generate_signature_hmac_sha256(secret:str, payload:dict):
    """
    Generates an HMAC SHA256 signature using Python's standard libraries.
    The payload must be a byte string.
    """
    payload_json = json.dumps(payload, separators=(",", ":"))
    signature = hmac.new(
        secret.encode(),
        payload_json.encode(),
        hashlib.sha256
    ).hexdigest()
    return f"{signature}"

@app.post("/webhookproxy")
async def webhookproxy(
    request: Request,
    final_dest_url: str | None = Query(
        default=None,
        alias="url",
        description="Optional query parameter to override the default webhook destination URL."
    )
):
    """
    This endpoint intercepts the webhook from your original app.
    It can now optionally take a query parameter to dynamically set the destination URL.
    1. Receives the webhook request.
    2. Generates an HMAC-SHA256 signature based on the request body.
    3. Adds the signature to a new header.
    4. Forwards the request to the final destination.
    """
    # Determine the destination URL to use for this request.
    destination_url_to_use = final_dest_url if final_dest_url else WEBHOOK_DESTINATION

    if not destination_url_to_use:
        raise HTTPException(status_code=400, detail="No webhook destination URL provided.")

    try:
        # 1. Get the raw body of the request from the original app.
        payload = await request.json()
        payload_bytes = await request.body()
        
        # 2. Generate the signature using the robust HMAC-SHA256 algorithm.
        signature = generate_signature_hmac_sha256(WEBHOOK_SECRET, payload)
        print(f"Generated signature.")
    except Exception as e:
        print(f"Error generating signature: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate signature")

    # 3. Create new headers, including the original headers and the new signature.
    # FastAPI's request.headers is a special type, so we convert it to a dict first.
    forward_headers = dict(request.headers)
    forward_headers["X-Signature"] = signature

    # 4. Forward the request to the final destination.
    print(f"Forwarding request to: {destination_url_to_use}")
    try:
        response = requests.post(
            destination_url_to_use,
            data=payload_bytes,
            headers=forward_headers,
            timeout=30
        )
        print(f"Destination server response status: {response.status_code}")
        
        # 5. Return the response from the destination server back to the original app.
        # We need to return a FastAPI Response object, not a requests Response object.
        return Response(content=response.content, status_code=response.status_code, media_type=response.headers.get("content-type"))
    except requests.exceptions.RequestException as e:
        print(f"Error forwarding webhook: {e}")
        raise HTTPException(status_code=502, detail="Failed to forward webhook")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
