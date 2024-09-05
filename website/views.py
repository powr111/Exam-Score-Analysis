from flask import Blueprint, render_template, Flask
from .analysis import df_html, aggragations_associates, aggragations_highschool, test, decide, accuracy # ignore import error it works!

#blueprints for the routes
views = Blueprint('views', __name__)

@views.route('/')
def home():
    return render_template("home.html")

@views.route('/about')
def about():
    return render_template("about.html")

@views.route('/descriptive')
def descr():
    return render_template("descriptive.html",associates_mean = aggragations_associates['mean'],highschool_mean = aggragations_highschool['mean'])

@views.route('/predictive')
def pred():
    return render_template("predictive.html", accuracy = accuracy)

@views.route('/diagnostic')
def diag():
    return render_template("diagnostic.html", test_val = round(test[0],4), test_p = round(test[1],4), decide = round(decide,4))

@views.route('/analysis-types')
def types():
    return render_template("types.html", df_html = df_html)