from flask import Flask, render_template, request  
from bm25 import getDataFromJson, organize, retrieve 

app = Flask(__name__)  


@app.route("/", methods=["GET", "POST"])
def index():
    results = []  
    query = ""  
    if request.method == "POST":
        query = request.form["query"] 
        results = retrieve(query)  
    return render_template("index.html", query=query, results=results)


if __name__ == "__main__":
    if not os.path.exists("full_data_processed_FINAL.p"):
        print("Processing data...") 
        getDataFromJson() 
        organize() 

    app.run(debug=True)
