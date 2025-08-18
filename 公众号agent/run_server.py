import asyncio
import uvicorn
from pathlib import Path


def main():
    # Default: run FastAPI app on localhost:8000
    uvicorn.run(
        "modules.api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()


