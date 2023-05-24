import subprocess

class MLFLOW_SETUP():

    def __init__(self ,  SSH_HOST="data7-db1.cyverse.org" ,  SSH_PORT=5432 ,  SSH_KEY="ssh.key" ,  DATABASE='chest_db_v2' ,  DATABASE_USER_PASSWORD=1234 ,  ARTIFACT_DIR="/home/artinmajdi/mlflow_data/artifact_store"):
        """ _summary_

        Args:
            SSH_HOST (str, optional): _description_. Defaults to "data7-db1.cyverse.org".
            SSH_PORT (str, optional): _description_. Defaults to "5432".
            SSH_USER (str, optional): _description_. Defaults to "artinmajdi".
            SSH_KEY (str, optional): _description_. Defaults to "ssh.key".
            DATABASE_USER_PASSWORD (str, optional): _description_. Defaults to "password".
            artifact (str, optional): _description_. Defaults to "/home/artinmajdi/mlflow_data/artifact".



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

        self.VIEW_PORT = None
        self.ssh_session = None
        self.SSH_HOST     = SSH_HOST
        self.SSH_PORT     = SSH_PORT
        self.LOCAL_PORT   = 5000
        self.SSH_USER     = "artinmajdi"
        self.SSH_KEY      = SSH_KEY

        self.ARTIFACT_DIR = ARTIFACT_DIR
        self.DATABASE     = DATABASE
        self.DATABASE_USER_PASSWORD = DATABASE_USER_PASSWORD

    def ssh_tunneling(self):

        command     = f'ssh -N -L 5000:localhost:{self.SSH_PORT} {self.SSH_USER}@{self.SSH_HOST} &'

        self.ssh_session = subprocess.Popen('exec ' + command, stdout=subprocess.PIPE, shell=True)

        # wait until session process is finished
        self.ssh_session.wait()
        print("ssh tunnel is running")

        return self.ssh_session

    def mlflow_ui(self, VIEW_PORT=6789):

        self.VIEW_PORT = VIEW_PORT

        if subprocess.call(['nc', '-z', 'localhost', str(VIEW_PORT)]) == 0:
            print(f"MLFlow UI is already running on localhost:{VIEW_PORT}")
            return None

        command = f'mlflow ui --backend-store-uri postgresql://{self.SSH_USER}:{self.DATABASE_USER_PASSWORD}@localhost:{self.LOCAL_PORT}/{self.DATABASE} --port {VIEW_PORT}'
        return subprocess.Popen(f'exec {command}', stdout=subprocess.PIPE, shell=True)


if __name__ == '__main__':
    mlflow_setup = MLFLOW_SETUP( )
    mlflow_setup.ssh_tunneling()
    p = mlflow_setup.mlflow_ui()
