# Installation

```bash
    >> conda create -n <env_name>
    >> conda activate <env_name>
    >> conda install -c anaconda psycopg2 git
    >> conda install -c anaconda notebook
    >> pip install mlflow==1.12.1
    >> pip install pysftp
```

# Remote ssh to atmosphere server
```python
>> ssh -L -N {LOCAL_PORT}:localhost:{SSH_PORT} -i {SSH_KEY} {SSH_USER}@{SSH_HOST} &

SSH_HOST: str   <=>   Hostname or IP address of the machine to tunnel to.
SSH_PORT: int   <=>   Port on the remote machine to tunnel to.
SSH_USER: str   <=>   Username for SSH authentication.
SSH_KEY : str   <=>   Path to the SSH private key file.
LOCAL_PORT : int, optional   ->   Local port to bind to.
```

Returns
-------
tunnel : SSHClient
    SSH client used to create the tunnel.


Command (in CLI)
-------


Saving the ssh credentials (in CLI)
--------------------------
    > ssh-keygen -t rsa
    > ssh-copy-id -i {SSH_KEY}  {SSH_USER}@{SSH_HOST}


Killing the ssh-tunnel (in SCRIPT)
----------------------
    > ssh_session.kill()

```bash
    >> ssh -L 5000:localhost:5432 artinmajdi@data7-db1.cyverse.org # -p 1657

    >> ssh artinmajdi@data7-db1.cyverse.org -p 22 -D 5000

    >> ssh -L <local-port,5000>:<remote-ip>:<postgres-port,5432> <usernname>@<remote-ip>
```

## 1.3.Viewing the results in mlflow server

Remote postgres server:
```bash
    >> mlflow ui --backend-store-uri postgresql://mlflow_developer:1234@localhost:5000/>> mlflow_db --port 6789
    >> mlflow ui --backend-store-uri postgresql://artinmajdi:1234@localhost:5000/phyto_oracle_db --port 6789
```

Local postgres server:

```bash
    >> mlflow ui --backend-store-uri postgresql://mlflow_developer:1234@localhost:5432/>> mlflow_db --port 6789
```

## 1.4.Runing the code

### 1.4.1.Directly from GitHub

 -v specify the GitHub Branch

```bash
    >> export MLFLOW_TRACKING_URI=http://127.0.0.1:6789

    >> mlflow experiments list

    >> mlflow run --no-conda --experiment-id experiment_id -P epoch=2 https://github.com/artinmajdi/mlflow_workflow.git -v main

    >> mlflow run code --no-conda --experiment-id experiment_id -P epoch=2
```

## 1.5.viewing the outputs

### 1.5.1.UI with postgres

REMOTE postgres server:
Step 1: Connecting to remote server through ssh tunneling

Step 2: Connecting to remote postgres server

```bash
    >> mlflow ui --backend-store-uri postgresql://mlflow_developer:1234@localhost:5000/>> mlflow_db --port 6789
```

LOCAL postgres server:

```bash
    >> mlflow ui --backend-store-uri postgresql://mlflow_developer:1234@localhost:5432/>> mlflow_db --port 6789
```

### 1.5.2.Show experiments/runs list

```bash
    >> export MLFLOW_TRACKING_URI=http://127.0.0.1:{port}
    >> mlflow runs list --experiment-id <id>
```

# sftp

source: <https://public.confluence.arizona.edu/display/UAHPC/Transferring+Files#TransferringFiles-GeneralFileTransfers>

step 0: Save the ssh authentication credentials
step 1: ```sftp://username@filexfer.hpc.arizona.edu:/home/u29/mohammadsmajdi/projects/mlflow/artifact_store```

-----------------------------------------------------------------------------

# MLflow Tracking

MLflow is an open source platform for managing the end-to-end machine learning lifecycle.

It has the following primary components:

    Tracking:       record experiments to compare parameters and results.
    Models:         manage and deploy models from a variety of ML libraries to a variety of model serving and inference platforms.
    Projects:       package ML code in a reproducible form to share with other data scientists or transfer to production.
    Model Registry: Allows you to centralize a model store for managing models’ full lifecycle stage transitions: from staging to production, with capabilities for versioning and annotating.
    Model Serving:  Allows you to host MLflow Models as REST endpoints.

Languages:    Java, Python, R, and REST APIs
Frameworks:   Keras , Scikit-learn , PySpark , XGBoost , TensorFlow , PyTorch , SpaCy , Fastai

```mlflow.framework.autolog()```

can be used to automatically capture  model-specific metrics, parameters, and model artifacts

- Metrics & Parameters:\

    Training loss; validation loss; user-specified metrics

    Metrics associated with the EarlyStopping callbacks (restore_best_weight, last_epoch), learning rate; epsilon

- Artifacts \
    Model summary on training start \
    MLflow Model (Keras model) on training end

Manual logging:

```python
    >> mlflow.log_metric("test_acc", acc=0.9)

    >> mlflow.log_param("epochs", epoch=100)

    >> mlflow.<framework>.log_model(model, "myModel")

    >> mlflow.log_artifact("absolute path to file", "artifact name")
```

Two types of experiments:

1. workspace: can be created from the Workspace UI or the MLflow API.
2. notebook

Active experiment, can be set using:

1. ```mlflow.set_experiment()```
2. ```mlflow.start_run(run_id=None, experiment_id=None)```

Artifact Stores: suitable for large data (such as an S3 bucket), log models etc.
    Amazon S3 and S3-compatible storage , Azure Blob Storage  , Google Cloud Storage  , FTP server , SFTP Server , NFS , HDFS (Apache Hadoop)

#### Where Runs Are Recorded

To log runs remotely, set the MLFLOW_TRACKING_URI environment variable to a tracking server’s URI or call mlflow.set_tracking_uri().

- Local
- Database encoded as ```<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>```.
  - dialects: mysql, mssql, sqlite, and postgresql
- HTTP server (specified as <https://my-server:5000>), that hosts an MLflow tracking server.
- Databricks

# MLflow Projects: MLProject.yaml

A convention for organizing your code to let other data scientists run it

Each project is:

1) a directory of files
2) Git repository.

These APIs also allow submitting the project for remote execution

- Databricks
- Kubernetes

#### mlflow experiments

<https://www.mlflow.org/docs/latest/cli.html#mlflow-experiments>

# Configuring databricks

-> <https://docs.databricks.com/dev-tools/cli/index.html>

```pip install databricks-cli```

#### run databricks configure

Host: <https://community.cloud.databricks.com>
User:
Pass:
Repeat pass:

# setting the tracking server

LOCAL-DISC: saving the runs on a designated uri

```python
    mlflow.set_tracking_uri('file:/home/petra/mlrun_store')
```

LOCAL-DATABASE:

```python
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
```

REMOTE-DATABASE:

```python
    mlflow.set_tracking_uri("databricks")
```

# Backend store

Database (backend-store-uri) needs to be encoded as dialect+driver://username:password@host:port/database.

#### Style of backend store for postgres:

```bash
    --backend-store-uri <dialect_driver>://<username>:<password>@<ip>/<database_name>'
```

Example (default):

```bash
    mlflow ui  --backend-store-uri <absolute-path-to-mlruns> \
               --default-artifact-root <absolute-path-ro-artifacts> \
               --host 0.0.0.0
               --port 5000
```

Example:

```bash
    >> mlflow ui --port 5000 \
    --backend-store-uri postgresql://mlflow_developer:1234@localhost/mlflow_db
    --backend-store-uri postgresql://mlflow_developer:1234@192.168.0.19/mlflow_db
```


HPC for artifact:

```bash
    --default-artifact-root sftp://mohammadsmajdi@filexfer.hpc.arizona.edu:/home/u29/mohammadsmajdi/projects/mlflow/artifact_store
```

Local storage for artifact:

```bash
    --default-artifact-root file:/Users/artinmac/Documents/Research/Data7/mlflow/artifact_store
```


Example to HPC-artifact & remote-postgres

```bash
    >> mlflow server --backend-store-uri postgresql://mlflow_developer:1234@128.196.142.23/mlflow_db --default-artifact-root sftp://mohammadsmajdi@filexfer.hpc.arizona.edu:/home/u29/mohammadsmajdi/projects/mlflow/artifact_store --port 5000
```

## Access to remote server postgres

Source: <https://www.thegeekstuff.com/2014/02/enable-remote-postgresql-connection/>

### Step 0

#### Completely purging postgresql

1. via ```apt-get```

1.1 Checking postgresql packages

```bash
    >> dpkg -l  grep postgres
```

1.2 Uninstalling all installed packages

```bash
    >> sudo apt-get --purge remove pgdg-keyring postgresql*
```

2. via brew

```bash
    >> brew uninstall postgresql
    >> rm .psql_history
```

#### Installing postgres

1. via apt-get

```bash
    >> sudo apt-get install postgresql postgresql-contrib postgresql-server-dev-all
```

2. via brew

2.1 Installing Brew in Ubuntu

```bash
    >> sudo apt-get install build-essential curl file
    >> sudo apt install linuxbrew-wrapper
    >> brew
```

2.2 Installing postgresql using brew

```bash
    >> brew install postgresql
```

2.3 Conda Environment

```bash
    >> conda create -n mlflow python=3.7
    >> conda install -c conda-forge scikit-learn scikit-image nibabel nilearn matplotlib numpy ipython pandas
    >> conda install -c anaconda tensorflow>=2.0 keras
```

#### Check PostgreSQL version

```bash
    >> apt show postgresql
```

### Setting up remote/client access

#### Find the remote server ip

Ubuntu:

```bash
    >> ip addr show
```

MacOS:

```bash
    >> ipconfig getifaddr en0
```

#### Set up server to listen to clients (postgresql.conf & pg_hba.conf)

Add the line: host  'all   all  \<client-ip\>/24   trust'

Ubuntu: ```vim /home/linuxbrew/.linuxbrew/var/postgres/pg_hba.conf```
Ubuntu: ```vim /etc/postgresql/9.1/main/pg_hba.conf```
Macos:  ```vim /usr/local/var/postgres/pg_hba.conf```

Change the postgresql.conf on server machine to listen to all addresses.

Ubuntu: ```vim /home/linuxbrew/.linuxbrew/var/postgres/postgresql.conf```
MacOS:  ```vim /usr/local/var/postgres/postgresql.conf```
-  Change ```listen_addresses=localhost``` to ```listen_addresses=*```

#### Restart postgres on your server

Ubuntu:

```bash
    >> pg_ctl -D /home/linuxbrew/.linuxbrew/var/postgres stop
    >> pg_ctl -D /home/linuxbrew/.linuxbrew/var/postgres start

    >> systemctl stop postgresql
    >> systemctl start postgresql
    >> systemctl status postgresql
```


MacOS:

```bash
    >> pg_ctl -D /usr/local/var/postgres stop
    >> pg_ctl -D /usr/local/var/postgres start
    Or
    >> brew services restart postgresql

    >> psql postgres -h <remote-ip> -p 5432 -U mlflow_developer
```

#### Setting up postgres

Go into psql

    Sign in with the current signed in user (default)  >> psql postgres
    Go into a specific database & a specific user:     >> psql -d mlflow_db -U mlflow_user

    Show Users:                           \du
    list all databases                    \l
    Connect to a specific database:       \c database_name;
    list all tables in current database   \dt   or \dt+ (for more information)
    Show tables within the database       \d+
    Get detailed information on a table:  \d+ table_name
    Showing information on database name, username, port, socket path    \conninfo

    Create a new role with a username and password: CREATE ROLE username NOINHERIT LOGIN PASSWORD password;

    Query all data from a table:                    SELECT * FROM table_name;

    Create a new database:                          CREATE DATABASE [IF NOT EXISTS] <database_name>;
    Delete a database permanently:                  DROP DATABASE [IF EXISTS] <database_name>;

    create a user                                   CREATE USER <username> WITH ENCRYPTED PASSWORD <password';
    delete a user                                   DROP USER <username>

    grant access to a database                      GRANT <privilege_list or ALL>  ON  <table_name> TO  <username>;
    grant access to a database                      GRANT <privilege_list or ALL>  ON  DATABASE <database_name> TO  <username>;

## Restoring and backing up postgresql databases

[source](https://www.postgresql.org/docs/current/backup-dump.html)

The idea behind this dump method is to generate a file with SQL commands that, when fed back to the server, will recreate the database in the same state as it was at the time of the dump. PostgreSQL provides the utility program pg_dump for this purpose. The basic usage of this command is:

        >> pg_dump dbname > dumpfile


Create a new database:

    >> CREATE DATABASE [IF NOT EXISTS] <database_name>;

Restore a database from a dump file

    >> psql dbname < dumpfile


# iRODS from HPC to CyVerse

## Connecting

```bash
    >> iinit
    >> server address:  data.cyverse.org
    >> port number:     1247
    >> iRODS user name: <CyVerse-username>
    >> iRODS zone:      iplant
    >> iRODS password:  <password>
```

## iCommands

```bash
    >> ils , icd , ipwd
    >> iput , iget
```

## Accessing CyVerse profile/data through web browser

<https://data.cyverse.org/dav/iplant/home/artinmajdi/>

## IMPORTANT: Reading data through iRODS from within python

To be able to have datasets in one place and run the HPC instances from that place automatically

<https://github.com/irods/python-irodsclient>

# Docker

```bash
    >> docker pull nvidia/cuda
```

## see ports that are listening

### Mac

```bash
    sudo lsof -i -n -P | grep TCP
```

## from train.py

 REMOTE postgres server:
    Step 1 (before running the code): Connecting to remote server through ssh tunneling

    Step 2 (after running the code): Connecting to remote postgres server
        mlflow ui --backend-store-uri postgresql://artinmajdi:1234@localhost:5000/mnist_db --port 6789

    Run from github:
        export MLFLOW_TRACKING_URI=http://127.0.0.1:{port} # port: 6789 or 5000
        mlflow run --no-conda --experiment-id experiment_id -P epoch=2 https://github.com/artinmajdi/mlflow_workflow.git -v main

        mlflow ui --backend-store-uri postgresql://mlflow_developer:1234@localhost:5000/mlflow_db  --default-artifact-root sftp://artinmajdi:<password>!@128.196.142.17:/home/artinmajdi/mlflow/artifact_store --port 6789

    Map network drive (CyVerse data storage) on macOS
        1. In Finder, either hit Command+K to bring up “Connect to Server” or click Go > Connect to Server.
        2. Enter login details and password. https://data.cyverse.org/dav/iplant/home/artinmajdi

## Sign-in

Sign in with the current signed-in user (default)

    >> psql -U postgres
    >> password: 1234

Go into a specific database & a specific user:

    psql -d mlflow_db -U mlflow_user
    psql postgres -h 128.196.65.115 -p 1657 -U artinmajdi

## List users/databases/info

Show Users:

    \du

list all databases

    \l

## Creating new user (a.k.a. role)

Create a new role with a username and password:

    CREATE USER <username> WITH ENCRYPTED PASSWORD <password> LOGIN SUPERUSER;

Delete a user

    DROP USER <username>

### Grant Access to a table/database

Grant access to a database

    GRANT <privilege_list or ALL>  ON  DATABASE <database_name> TO  <username>;

Grant access to a table

    GRANT <privilege_list or ALL>  ON  <table_name> TO  <username>;

Revoke all privileges

    REVOKE ALL PRIVILEGES on DATABASE <database_name> FROM <user>;

## Creating/removing new database

Create a new database:

    CREATE DATABASE [IF NOT EXISTS] <database_name>;

Delete a database permanently:

    DROP DATABASE [IF EXISTS] <database_name>;

vim databse

    ALTER DATABASE name RENAME TO new_name

Change ownership

    ALTER DATABASE name OWNER TO new_owner

## Viewing tables

Connect to a specific database:

    \c database_name;

Showing information on database name, username, port, socket path

    \conninfo

list all tables in current database

    \dt   or \dt+ (for more information)

Get detailed information on a table:

    \d+ table_name

View all data inside a table:

    SELECT * FROM table_name;

## How to get TLS version

openssl ciphers -v | awk '{print $2}' | sort | uniq

## to see the ports used in ubuntu

netstat -tuanlp

## sftp

sftp -oPort=1657 bartinmajdi@data7-db1.cyverse.org/home/artinmajdi/mlflow_data/artifact_store
