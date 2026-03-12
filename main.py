from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def index():
    """
    Basic function in the app
    """
    return {"data": {"name": "Hello World!"}}


@app.get("/about")
def about():
    """About response"""
    return {"data": {"name": "About Page"}}


if __name__ == "__main__":
    index()
