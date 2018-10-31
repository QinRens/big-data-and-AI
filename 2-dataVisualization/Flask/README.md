# Dashboard

Flask + ZingCharts.js + Python + Webpage

+ Parent templates:
	- layout.html – navigator & header & footer of the introduction webpage
		- Link to the QinRen main page
	- layout2.html – navigator & header of the other webpages
+ Child webpages:
	- intro.html – introduction page of the dashboard
		- Link to every child page
	- dataPro.html – data preprocessing, upload file, set parameters and visualization
		- Link to the previous one(intro.html)
		- When the data preprocessing finished, it can link to the next one
	- featureEng.html – feature engineering
		- Same as the previous one
	- ml.html – machine learning
	- help.html – show the details about how to use the dashboard correctly	 
