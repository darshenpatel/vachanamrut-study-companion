from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Test server working"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/themes/")
async def themes():
    return ["devotion", "faith", "surrender", "service", "knowledge"]