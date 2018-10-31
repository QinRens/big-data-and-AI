from flask import Flask
from flask import render_template
from flask import request, session, redirect, url_for,make_response
from werkzeug import secure_filename
import time
import pandas as pd
import os
import json
import numpy as np
from datetime import datetime
import re
import paramiko
import ssh
from contextlib import closing
import scpclient 

# #f45622 0%, #f53e54 100%);
# http://www.bjhee.com/flask-6.html
app = Flask(__name__)
 

 # Basic part of the html
@app.route('/')
@app.route('/into')
def intro():
	return render_template('intro.html')

@app.route('/dataPro')
def dataPro():
	return render_template('dataPro.html')

@app.route('/featureEng')
def featureEng():
	return render_template('featureEng.html')

@app.route('/ml')
def ml():
	return render_template('ml.html')

@app.route('/help')
def help():
	return render_template('help.html')

@app.route('/analyzeData2', methods=['GET', 'POST'])
def analyzeData2():
	if request.method == 'POST':
		f =request.files['file']
		f.save(secure_filename(f.filename))
		databank = pd.read_csv(f.filename)
		USA = databank[databank['Country Code'] == 'USA']
		USAData = USA.iloc[:, 4:]
		USADiscr = USA.iloc[:, :4]
		USAData.index = USADiscr['Series Code']
		missingColumns = np.sum(USAData=='..',axis=0)
		missingRows = np.sum(USAData=='..', axis=1)
		featureName = [str(i) for i in missingRows.index]
		missingRate = missingRows.values/float(USAData.shape[1])*100
		missingCount = np.sum(missingRows.values)
		allCount = USAData.shape[0]*USAData.shape[1]
		threshold = request.form.get('threshold', '70')
		fixType = request.form.get('fixing', 'linear')
		data = {'missingRate': list(missingRate), 'featureName':featureName, 'missingCount': missingCount, 'allCount': allCount, 
		'missingDateName':list(missingColumns.index), 'missingDateCount':list(missingColumns), 'threshold':int(threshold)} 
		return render_template('featureEng.html', name=f.filename, data=data)

@app.route('/analyzeData', methods=['GET', 'POST'])
def analyzeData():
	if request.method == 'POST':
		f =request.files['file']
		f.save(secure_filename(f.filename))
		featureData = pd.read_csv(f.filename, index_col=0)

		missingColumns = np.sum(featureData.isnull(), axis=0)
		missingRows = np.sum(featureData.isnull(), axis=1)
		featureName = [str(i) for i in missingColumns.index]
		missingRate = missingColumns.values/float(featureData.shape[0])*100
		missingCount = np.sum(missingColumns.values)
		allCount = featureData.shape[0]*featureData.shape[1]
		threshold = request.form.get('threshold', '70')
		fixType = request.form.get('fixing', 'linear')
		data = {'missingRate': list(missingRate), 'featureName':featureName, 'missingCount': missingCount, 'allCount': allCount, 
		'missingDateName':list(missingRows.index), 'missingDateCount':list(missingRows), 'threshold':int(threshold)} 
		return render_template('dataPro.html', name=f.filename, data=data)

@app.route('/linux', methods=['GET', 'POST'])
def linux():
	if request.method == 'POST':
		f =request.files['file']
		f.save(secure_filename(f.filename))
		# Connect the linux with ssh
		ssh = paramiko.SSHClient()
		# Load or generate the host key 
		ssh.load_system_host_keys()
		ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		# Connect the server
		ssh.connect("58.206.100.172", port=2222, username="username", password="password")
		# stdin,stdout,stderr = ssh.exec_command('mkdir aliServer')

		with closing(scpclient.Write(ssh.get_transport(), remote_path="/home/keyyd/aliServer")) as scp:
			scp.send_file(f.filename, preserve_times=True)
		stdin, stdout, stderr = ssh.exec_command('python aliServer/main.py')
		return render_template('help.html', name=f.filename, data=stdout.read())


# Test the upload file from dataPro.html
@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f =request.files['file']
		f.save(secure_filename(f.filename))
		df = pd.read_csv(f.filename).drop('Open', axis=1)
		chart_data = df.to_dict(orient='records')
		chart_data = json.dumps(chart_data, indent=2)
		data = {'chart_data': chart_data}

		sp500 = pd.read_csv(f.filename, index_col=0).drop('Open', axis=1)
		ZingData = [[int(1000*time.mktime(time.strptime(str(sp500.index[i]), "%Y-%m-%d"))), sp500.ix[i, 'Close']] for i in range(sp500.shape[0])]

		return render_template('dataPro.html', name=f.filename, data=data, ZingData =ZingData)

@app.route("/layout/", methods=['POST'])
def move_forward():
    #Moving forward code
    forward_message = "Moving Forward..."
    print forward_message
    return render_template('Resume.html');

@app.route('/resume')
def resume():
	return render_template('Resume.html')

if __name__ == '__main__':
    app.run(debug=True)




