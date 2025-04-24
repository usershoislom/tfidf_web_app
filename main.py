import uvicorn
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.utils import calculate_tfidf

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, files: list[UploadFile] = File(...)):
    corpus_texts = []

    for f in files:
        contents = await f.read()
        corpus_texts.append(contents.decode("utf-8"))

    if not corpus_texts:
        return templates.TemplateResponse(
            "index.html", {"request": request, "table": []}
        )

    target_text = corpus_texts[0]
    other_texts = corpus_texts[1:] if len(corpus_texts) > 1 else [target_text]

    df = calculate_tfidf(target_text, other_texts)
    return templates.TemplateResponse(
        "index.html", {"request": request, "table": df.to_dict(orient="records")}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
