from flask import Flask

def make_app():
    
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.config['SECRET_KEY'] = 'ThE_H09E_0F_4_J0b'

    from .views import views
    app.register_blueprint(views,url_prefix='/')

    return app