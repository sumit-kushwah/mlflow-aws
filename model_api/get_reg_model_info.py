from email.policy import default
import mlflow
from mlflow.tracking import MlflowClient
import click
from pprint import pprint
import boto3
import os

@click.command(help="This script is to get the registered model info from mlflow tracking server")
@click.option("--model_name", default='salary_predictor', help="Model name under which model has been registered")
@click.option("--tracking_uri", default="http://52.205.59.55/", help="Mlflow remote tracking server uri")
@click.option("--get_model", default=False, help="Do you want to download model file from S3? (True/False)")
def get_reg_model_info(model_name, tracking_uri, get_model):
	mlflow.set_tracking_uri(tracking_uri)
	client = MlflowClient()
	str_to_search = "name='"+ model_name + "'"
	current_stage = 'Production'
	latest_creation_timestamp = 0
	latest_model_info = {}
	for mv in client.search_model_versions(str_to_search):
		model_info = dict(mv)
		if model_info['current_stage']==current_stage:
			if model_info['creation_timestamp'] > latest_creation_timestamp:
				latest_creation_timestamp = model_info['creation_timestamp']
				latest_model_info = model_info
	pprint(latest_model_info, indent=4)
	# s3://banned-keyword-ml-model/0/239d1e6dfd2b4ef9a2df3f74fafab48c/artifacts/fasttext_model/model.fasttext
	if get_model:
		s3_resource = boto3.resource('s3')
		my_bucket = s3_resource.Bucket('banned-keyword-ml-model')
		model_path = latest_model_info['source'].split('/')
		prefix = '/'.join(model_path[3:-1])
		print(prefix)
		objects = my_bucket.objects.filter(Prefix=prefix)
		for obj in objects:
			path, filename = os.path.split(obj.key)
			if filename=='model.fasttext':
				my_bucket.download_file(obj.key, model_name + '_' + filename + '_version_' + latest_model_info['version'])

if __name__ == '__main__':
    get_reg_model_info()
