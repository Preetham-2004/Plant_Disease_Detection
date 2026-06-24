from pathlib import Path

import modal


APP_NAME = "plantguard-backend"
MODEL_MOUNT_PATH = Path("/models")
UPLOADS_MOUNT_PATH = Path("/uploads")

model_volume = modal.Volume.from_name("plantguard-models", create_if_missing=True)
uploads_volume = modal.Volume.from_name("plantguard-uploads", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("website.backend")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("plantguard-backend")],
    volumes={
        str(MODEL_MOUNT_PATH): model_volume,
        str(UPLOADS_MOUNT_PATH): uploads_volume,
    },
    timeout=900,
)
@modal.asgi_app()
def fastapi_app():
    import os

    os.environ["PLANTGUARD_MODEL_ROOT"] = str(MODEL_MOUNT_PATH)
    os.environ["PLANTGUARD_UPLOADS_DIR"] = str(UPLOADS_MOUNT_PATH)

    from website.backend.app.main import app as web_app

    return web_app
