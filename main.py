import joblib
from typing import Annotated
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/model", StaticFiles(directory="model"), name="model")


@app.get("/", response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )


@app.post("/get-prediction/")
def make_prediction(text: Annotated[str, Form()]):
    model = joblib.load('model/model.pkl')
    # Function to make predictions using the loaded model
    intent = model.predict([text])[0]
    print(f"Intent: {intent}")
    return {
        "question": text,
        "intention": intent
    }