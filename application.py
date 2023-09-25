from flask import Flask,render_template,request
import pickle
import numpy as np

model  = None

if model == None:
    
    model=pickle.load(open('diabetes.pkl','rb'))
    
application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result',methods=['GET','POST'])
def predict():
    preg = float(request.form.get("preg"))
    glu = float(request.form.get("glu"))
    sk = float(request.form.get("sk"))
    ins = float(request.form.get("in"))
    bmi = float(request.form.get("bmi"))
    dpf = float(request.form.get("dpf"))
    age = float(request.form.get("age"))
    bp = float(request.form.get("bp"))
    
    result = model.predict(np.array([preg,glu,bp,sk,ins,bmi,dpf,age]).reshape(1,-1))
    
    if result[0] == 1:
        result = 'Suffered From diabetes.'
        
    else:
        result = 'Not Suffered From diabetes.'
        
    return render_template('index.html',result = result)
    
    
    
    



if __name__=='__main__':
    app.run(debug=True)