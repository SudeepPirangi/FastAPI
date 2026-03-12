from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
def index():
    """
    Basic function in the app
    """
    return {"data": "Blog list"}


@app.get("/blogs")
def get_all_blogs(limit: int = 10, published: bool = True):
    """
    Get List of all blogs
    """
    return {"data": f"List of {limit} blogs - {published}"}


class Blog(BaseModel):
    """Blog model"""

    title: str
    body: str
    published: Optional[bool]


@app.post("/blog")
def create_blog(request: Blog):
    """Create a Blog"""
    return request


@app.get("/blog/{blog_id}")
def show_blog(blog_id: int, published: bool = True):
    """Individual Blog"""
    return {"data": f"Blog #{blog_id}, published - {published}"}


if __name__ == "__main__":
    index()
