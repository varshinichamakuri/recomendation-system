import os

from flask import Flask, render_template, request

from popularity_recommender import PopularityRecommender

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ml-100k")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        top_n = int(request.form.get("top_n", 10))
        if top_n < 1:
            top_n = 10

        recommender = PopularityRecommender(data_dir=DATA_DIR)
        recommender.fit(quantile_threshold=0.90)
        recommendations = recommender.recommend(n=top_n).copy()
        recommendations["weighted_score"] = recommendations["weighted_score"].round(3)
        recommendations["vote_average"] = recommendations["vote_average"].round(2)

        return render_template(
            "results.html",
            movies=recommendations.to_dict("records"),
            top_n=top_n,
        )
    except Exception as exc:
        return render_template("index.html", error=str(exc)), 400


if __name__ == "__main__":
    app.run(debug=True)
