import uvicorn
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.utils import calcutate_tfidf
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    text = contents.decode("utf-8")
    df = calcutate_tfidf(text)
    return templates.TemplateResponse("index.html", {"request": request, "table": df.to_dict(orient="records")})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)