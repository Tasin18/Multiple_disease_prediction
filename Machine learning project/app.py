from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__ , template_folder='templates', static_folder='static')

#load the pickle model
heart_disease_free = pickle.load(open('heart_disease_free.pkl', 'rb'))
heart_disease_premium = pickle.load(open('heart_disease_premium.pkl','rb'))


@app.route('/')
def Home():
    return render_template("index.html")

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/Heart_disease')
def heart_disease():
    return render_template("Heart_disease.html")

@app.route('/heart_premium')
def heart_premium():
    return render_template("heart_premium.html")

@app.route('/diabetes_free')
def diabetes_free():
    return render_template('diabetes_free.html')

@app.route('/diabetes_premium')
def diabetes_premium():
    return render_template('diabetes_premium.html')

@app.route('/breast_cancer_free')
def breast_cancer_free():
    return render_template('breast_cancer_free.html')

@app.route('/breast_cancer_premium')
def breast_cancer_premium():
    return render_template('breast_cancer_premium.html')

@app.route("/heart_disease_predict", methods = ['POST','GET'])
def predict():
    #get values from form
    sex = int(request.form['sex'])
    age = float(request.form['age'])
    cp = int(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])
    final = [np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])]
    prediction = heart_disease_free.predict(final)
    if(prediction[0]==1):
        return render_template('heart_disease_found.html')
    else:
        return render_template('heart_disease_not_found.html')


@app.route("/heart_premium_predict", methods = ['POST','GET'])
def predict_premium():
    #get values from form
    sex = int(request.form['sex'])
    age = float(request.form['age'])
    cp = int(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])
    final = [np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])]
    prediction = heart_disease_premium.predict(final)
    if(prediction[0]==1):
        return render_template('heart_disease_found.html')
    else:
        return render_template('heart_disease_not_found.html')
    
@app.route("/diabetes_free_predict", methods = ['POST','GET'])
def diabetes_free_predict():
    diabetes_free = pickle.load(open('diabetes_free.pkl','rb'))
    #get values from form
    sex = int(request.form['sex'])
    age = int(request.form['age'])
    urea = float(request.form['urea'])
    Cr = int(request.form['Cr'])
    chol = float(request.form['chol'])
    HbA1c = float(request.form['Hb'])
    TG = float(request.form['TG'])
    HDL = float(request.form['HDL'])
    LDL = float(request.form['LDL'])
    VLDL = float(request.form['VLDL'])
    BMI = float(request.form['BMI'])
    
    final = [np.array([sex,age,urea,Cr,HbA1c,chol,TG,HDL,LDL,VLDL,BMI])]
    prediction = diabetes_free.predict(final)
    if(prediction[0]==1):
        return render_template('diabetes_found.html')
    else:
        return render_template('diabetes_not_found.html')
 
@app.route("/diabetes_premium_predict", methods = ['POST','GET'])
def diabetes_premium_predict():
    diabetes_free = pickle.load(open('diabetes_premium.pkl','rb'))
    #get values from form
    sex = int(request.form['sex'])
    age = int(request.form['age'])
    urea = float(request.form['urea'])
    Cr = int(request.form['Cr'])
    chol = float(request.form['chol'])
    HbA1c = float(request.form['Hb'])
    TG = float(request.form['TG'])
    HDL = float(request.form['HDL'])
    LDL = float(request.form['LDL'])
    VLDL = float(request.form['VLDL'])
    BMI = float(request.form['BMI'])
    
    final = [np.array([sex,age,urea,Cr,HbA1c,chol,TG,HDL,LDL,VLDL,BMI])]
    prediction = diabetes_free.predict(final)
    if(prediction[0]==1):
        return render_template('diabetes_found.html')
    else:
        return render_template('diabetes_not_found.html')
    
@app.route("/cancer_free_predict", methods = ['POST','GET'])
def cancer_free_predict():
    diabetes_free = pickle.load(open('breast_cancer_free.pkl','rb'))
    #get values from form
    
    features = [float(x) for x in request.form.values()]
    final = [np.array(features)]
    prediction = diabetes_free.predict(final)
    if(prediction[0]==1):
        return render_template('cancer_found.html')
    else:
        return render_template('cancer_not_found.html')    


@app.route("/cancer_premium_predict", methods = ['POST','GET'])
def cancer_premium_predict():
    diabetes_free = pickle.load(open('breast_cancer_premium.pkl','rb'))
    #get values from form
    
    features = [float(x) for x in request.form.values()]
    final = [np.array(features)]
    prediction = diabetes_free.predict(final)
    if(prediction[0]==1):
        return render_template('cancer_found.html')
    else:
        return render_template('cancer_not_found.html')     
    
if __name__== '__main__':
    app.run(debug=True)

