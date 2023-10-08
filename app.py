from flask import Flask


app = Flask(__name__)


@app.route('/')
def hello_world():
    return '''<!doctype html>
<title>Hello from Flask</title>
{% if name %}
  <h1>Hello {{ name }}!</h1>
{% else %}
  <h1>Hello, World!</h1>'''
