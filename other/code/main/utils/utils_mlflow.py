# pylint: disable=too-few-public-methods
import os
import subprocess
import git
import mlflow

class MLFLOW_SETUP:

	def __init__(self, experiment_name='label_inter_dependence',  SSH_USER='artinmajdi', SSH_HOST="data7-db1.cyverse.org", SSH_PORT=5432, SSH_KEY="ssh.key", DATABASE='chest_db_v2', DATABASE_USER_PASSWORD=1234):
		""" _summary_

		Args:
			SSH_HOST (str, optional): _description_. Defaults to "data7-db1.cyverse.org".
			SSH_PORT (str, optional): _description_. Defaults to "5432".
			SSH_KEY (str, optional): _description_. Defaults to "ssh.key".
			DATABASE_USER_PASSWORD (str, optional): _description_. Defaults to "password".
			ARTIFACT_DIR (str, optional): _description_. Defaults to "/home/artinmajdi/mlflow_data/artifact".

		1. Creating an ssh-tunnel to server in the background

		1.1 Step1: Save the ssh credentials

			```python
				os.environ['USERNAME'] = "username"
				os.environ['PASSWORD'] = "password"
				os.environ['SSH_HOST'] = "ssh.host"
				os.environ['SSH_PORT'] = "ssh.port"
				os.eniron['SSH_USER'] = "ssh.user"
				os.environ['SSH_KEY'] = "ssh.key"
			```

			```bash
				#!/bin/bash
				export SSH_USER=<span style="color:red">USERNAME</span>
				export SSH_PASS=<span style="color:red">PASSWORD</span>
				export SSH_HOST=<span style="color:red">HOST</span>
				export SSH_PORT<span style="color:red">PORT</span>
				export SSH_KEY=<span style="color:red">KEY</span>
			```

			```bash
				> ssh-copy-id [username]@[server-ip]
				> ssh-keygen -t rsa
				> ssh-copy-id -i ~/.ssh/id_rsa.pub  user@server
			```

		1.2. Step2: Connect to the server in the background

			```bash
				> ssh -N -L 5000:localhost:5432 [username]@[server-ip]

				> export MLFLOW_TRACKING_URI=http://localhost:5000
				> export MLFLOW_TRACKING_USERNAME=username
				> export MLFLOW_TRACKING_PASSWORD=password
			```

		1.3 Killing the ssh tunnel

			```bash
				> pkill -f ssh
			```

			```python
							or
				subprocess.call(['pkill', '-f', 'ssh'])

							or
				# closing the child mlflow session
				mlflow.end_run()

				# closing the parent mlflow session
				mlflow.end_run()

				# closing the ssh session
				ssh_session.kill()
			```

		"""

		self.run = None
		self._client = None
		self.experiment_name = experiment_name
		self.ssh_session = None
		self._runID_list = {}

		self.SSH_HOST = SSH_HOST
		self.SSH_PORT = SSH_PORT
		self.LOCAL_PORT = 5000
		self.SSH_USER = SSH_USER
		self.SSH_KEY = SSH_KEY
		self.DATABASE = DATABASE
		self.DATABASE_USER_PASSWORD = DATABASE_USER_PASSWORD

	def do_ssh_tunnel(self):
		# Creates an SSH tunnel to the remote server, and then uses the tunnel to connect to the remote database
		command = f'ssh -N -L 5000:localhost:{self.SSH_PORT} {self.SSH_USER}@{self.SSH_HOST} &'
		ssh_session = subprocess.Popen('exec ' + command, stdout=subprocess.PIPE, shell=True)

		# wait for the ssh tunnel to be established
		while not ssh_session.communicate():
			pass

		return ssh_session

	def run_mlflow_ui(self, VIEW_PORT=6789):

		if subprocess.call(['nc', '-z', 'localhost', str(VIEW_PORT)]) == 0:
			print(f"MLFlow UI is already running on localhost:{VIEW_PORT}")
			return None

		# starts the MLFlow UI on the local machine
		command = f'mlflow ui --backend-store-uri postgresql://{self.SSH_USER}:{self.DATABASE_USER_PASSWORD}@localhost:{self.LOCAL_PORT}/{self.DATABASE} --port {VIEW_PORT}'
		return subprocess.Popen(f'exec {command}', stdout=subprocess.PIPE, shell=True)

	def set_up_experiment(self):

		# setting the tracking URI
		mlflow.set_tracking_uri(self.tracking_uri)

		# If the experiment doesn't exist, create it
		if not self.experiment_ID: 	mlflow.create_experiment(name=self.experiment_name, artifact_location=self.registry_uri)

		# Set the experiment
		mlflow.set_experiment(experiment_name=self.experiment_name)

	def download_artifacts(self, run_id=None, run_name=None ,  local_path='./', remote_path='', artifact_name=None):
		if run_name and (run_id is None):
			run_id = self.get_run(run_name=self.MLFlow_run_name).info.run_id

		os.makedirs(local_path, exist_ok=True)

		path = os.path.join(remote_path, artifact_name) if artifact_name else remote_path
		self.client.download_artifacts(run_id=run_id, dst_path=local_path, path=path)

	def get_run(self, run_name='tic-tac-toe', run_id=''): # type: (str, str) -> mlflow.entities.Run

		# Getting the runID from the run_name
		run_id = self.run_IDs_list[run_name] if len(run_id) > 0 else run_id

		# Returning the run correspondingto the run_id
		return mlflow.get_run(run_id)

	def run_setup(self, run_name=None, run_id=None, new_run=True, nested=False):

		if new_run:
			self.run = mlflow.start_run(run_name=run_name, nested=nested)
			mlflow.set_tag('mlflow.note.content', f'run_id: {self.run.info.run_id}')

		else:
			assert run_id or run_name, 'either run_id or run_name must be provided'
			run_id = self.run_IDs_list[run_name] if run_id is None else run_id
			self.run = mlflow.start_run(run_id=run_id, nested=nested)

		repo = git.Repo(search_parent_directories=True)
		mlflow.set_tag('mlflow.source.git.commit', repo.head.object.hexsha)
		mlflow.set_tag('mlflow.source.name', self.run.data.tags['mlflow.source.name'])

		return self.run

	@staticmethod
	def cleanup_mlflow_after_runs():
		while mlflow.active_run():
			mlflow.end_run()

	@property
	def client(self):
		if self._client is None:
			self._client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.registry_uri)
		return self._client

	@property
	def run_id(self):
		return self.run.info.run_id if hasattr(self, 'run') else None

	@property
	def experiment_ID(self):
		exp = self.client.get_experiment_by_name(self.experiment_name)
		return exp.experiment_id if exp is not None else None

	@property
	def run_IDs_list(self):
		get_runName = lambda run_info: mlflow.get_run(run_info.run_id).data.tags['mlflow.runName']

		if len(self._runID_list) == 0:
			self._runID_list = { get_runName(run_info): run_info.run_id for run_info in self.client.list_run_infos(self.experiment_ID)}

		return self._runID_list

	@property
	def registry_uri(self):
		ARTIFACT_DIR="/home/artinmajdi/mlflow_data/artifact_store"
		return f'sftp://{self.SSH_USER}@{self.SSH_HOST}:{ARTIFACT_DIR}'

	@property
	def tracking_uri(self):
		"""
		RUN UI with postgres and HPC:
		REMOTE postgres server:
			# connecting to remote server through ssh tunneling
			ssh -L 5000:localhost:5432 artinmajdi@data7-db1.cyverse.org

			# using the mapped port and localhost to view the data
			mlflow ui --backend-store-uri postgresql://artinmajdi:1234@localhost:5000/chest_db --port 6789

		RUN directly from GitHub or show experiments/runs list:

		export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

		mlflow runs list --experiment-id <id>

		mlflow run                 --no-conda --experiment-id 5 -P epoch=2 https://github.com/artinmajdi/mlflow_workflow.git -v main
		mlflow run mlflow_workflow --no-conda --experiment-id 5 -P epoch=2

		PostgreSQL server style
			server = f'{dialect_driver}://{username}:{password}@{ip}/{database_name}'

			postgres_connection_type:
				PORT, HOST = ('direct':     '5432', 'data7-db1.cyverse.org')
				PORT, HOST = ('ssh-tunnel': '5000', 'localhost')

			artifact = {
				'hpc':        f'sftp://mohammadsmajdi@filexfer.hpc.arizona.edu:/home/u29/mohammadsmajdi/projects/mlflow/artifact_store',
				'data7_db1':  f'sftp://artinmajdi@data7-db1.cyverse.org:/home/artinmajdi/mlflow_data/artifact_store'}

		"""

		server_view_mode='local' ,
		HOST = self.SSH_HOST if server_view_mode == 'remote' else 'localhost'
		PORT = self.SSH_PORT if server_view_mode == 'remote' else '5000'
		return f'postgresql://{self.SSH_USER}:{self.DATABASE_USER_PASSWORD}@{HOST}:{PORT}/{self.DATABASE}'

	def start_MLFlow(self, run_name='test', NEW_RUN=True, nested=False):

		print('Connecting to the server...')
		self.ssh_session = self.do_ssh_tunnel()

		print('Killing all active runs...')
		MLFLOW_SETUP.cleanup_mlflow_after_runs()

		print('Setting up the experiment...')
		self.set_up_experiment()

		print('Setting up the run...')
		self.run_setup(new_run=NEW_RUN, run_name=run_name, nested=nested)

	def kill_MLFlow(self):

		# closing the child mlflow session
		MLFLOW_SETUP.cleanup_mlflow_after_runs()

		# closing the ssh session
		self.ssh_session.kill()

class AIM1_3_MLFLOW_SETUP(MLFLOW_SETUP):

	def __init__(self, experiment_name='aim1_3_final_results'):

		MLFLOW_SETUP.__init__(self, SSH_HOST="data7-db1.cyverse.org", SSH_PORT=5432, SSH_KEY="ssh.key",
							  DATABASE='chest_db_v2', DATABASE_USER_PASSWORD=1234)

		self.do_ssh_tunnel()

		# sets the mlflow experiment
		self.set_up_experiment()
