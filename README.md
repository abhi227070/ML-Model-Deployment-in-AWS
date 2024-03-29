# Diabetes Prediction Deployment in AWS

## Table of Contents
1. [Project Overview](#project-overview)
2. [Use Cases](#use-cases)
3. [Project Structure](#project-structure)
4. [Running the Project](#running-the-project)
5. [Flask Code](#flask-code)
6. [Acknowledgements](#acknowledgements)
7. [License](#license)

## Project Overview

This project focuses on predicting diabetes using machine learning classification techniques while emphasizing MLOps (Machine Learning Operations) principles. The main objective is not only to build a predictive model but also to demonstrate how to deploy it on AWS (Amazon Web Services) cloud infrastructure with a CI/CD (Continuous Integration/Continuous Deployment) pipeline.

## Use Cases

1. **Automated Deployment**: Deploying the diabetes prediction model on AWS allows for automated deployment, ensuring that the latest version of the model is always available without manual intervention.
   
2. **Scalability**: Leveraging AWS infrastructure enables easy scalability of the prediction service to handle varying levels of demand, ensuring consistent performance even during peak times.
   
3. **Version Control**: Utilizing CI/CD pipelines ensures version control of the deployed model, enabling easy rollback to previous versions if necessary.
   
4. **Monitoring and Logging**: Implementing monitoring and logging mechanisms in AWS allows for real-time tracking of model performance, usage metrics, and any potential issues, ensuring reliability and performance optimization.

## Project Structure

The project consists of the following components:

1. **Machine Learning Model**: A machine learning model trained on diabetes dataset to predict the likelihood of an individual having diabetes based on various parameters.

2. **Flask UI**: A user interface created using Flask, HTML, and CSS to interact with the machine learning model. This UI allows users to input their parameters and receive a prediction for diabetes.

3. **AWS Infrastructure**: Deployment of the project on AWS involves creating an EC2 instance and uploading the project code. The application can be run on the AWS server using command-line commands.

## Running the Project

To run the project locally:
1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Flask application using `python application.py`.
4. Access the UI in your web browser at `http://localhost:5000`.

To deploy the project on AWS:
1. Create an EC2 instance on AWS and configure it.
2. Upload the project code to the EC2 instance.
3. Install the necessary dependencies on the EC2 instance.
4. Run the Flask application on the EC2 instance using command-line commands.
5. Access the deployed application using the public IP address of the EC2 instance.

## Flask Code

```python
from flask import Flask, render_template, request
import pickle
import numpy as np

model = None

if model == None:
    model = pickle.load(open('diabetes.pkl', 'rb'))

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/result', methods=['GET', 'POST'])
def predict():
    preg = float(request.form.get("preg"))
    glu = float(request.form.get("glu"))
    sk = float(request.form.get("sk"))
    ins = float(request.form.get("in"))
    bmi = float(request.form.get("bmi"))
    dpf = float(request.form.get("dpf"))
    age = float(request.form.get("age"))
    bp = float(request.form.get("bp"))
    
    result = model.predict(np.array([preg, glu, bp, sk, ins, bmi, dpf, age]).reshape(1, -1))
    
    if result[0] == 1:
        result = 'Suffered From diabetes.'
    else:
        result = 'Not Suffered From diabetes.'
        
    return render_template('input.html', result=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
```

## Acknowledgements

This project utilizes Flask for building the user interface and AWS for deployment infrastructure. It follows MLOps principles for model deployment and management.

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.
