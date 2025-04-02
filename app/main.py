from fastapi import FastAPI
from app.api import endpoints
import uvicorn

app = FastAPI()
app.include_router(endpoints.router)

if __name__ == "__main__":
    #uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)