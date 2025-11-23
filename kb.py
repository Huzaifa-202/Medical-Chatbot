Logs


Instances
Lookback period
Connected!
2025-11-23T12:13:54.3889323Z    _____
2025-11-23T12:13:54.3889599Z   /  _  \ __________ _________   ____
2025-11-23T12:13:54.3889629Z  /  /_\  \\___   /  |  \_  __ \_/ __ \
2025-11-23T12:13:54.3889649Z /    |    \/    /|  |  /|  | \/\  ___/
2025-11-23T12:13:54.388967Z \____|__  /_____ \____/ |__|    \___  >
2025-11-23T12:13:54.388969Z         \/      \/                  \/
2025-11-23T12:13:54.3889711Z A P P   S E R V I C E   O N   L I N U X
2025-11-23T12:13:54.3889729Z
2025-11-23T12:13:54.3889751Z Documentation    : http://aka.ms/webapp-linux
2025-11-23T12:13:54.3889773Z Python quickstart: https://aka.ms/python-qs
2025-11-23T12:13:54.3889793Z Python version   : 3.12.12
2025-11-23T12:13:54.3889841Z Instance Name    : lw1sdlwk000853
2025-11-23T12:13:54.388987Z Instance Id      : 91baf352954053b10aec40a76c08c5d8df4b45053890ea68ab5fc855af1343fa
2025-11-23T12:13:54.3889888Z
2025-11-23T12:13:54.388991Z Note: Any data outside '/home' is not persisted
2025-11-23T12:13:55.1362442Z Starting OpenBSD Secure Shell server: sshd.
2025-11-23T12:13:55.160032Z WEBSITES_INCLUDE_CLOUD_CERTS is not set to true.
2025-11-23T12:13:55.2378775Z Updating certificates in /etc/ssl/certs...
2025-11-23T12:14:26.7381315Z rehash: warning: skipping duplicate certificate in azl_Sectigo_Public_Server_Authentication_Root_R46.pem
2025-11-23T12:14:26.7824935Z rehash: warning: skipping duplicate certificate in azl_Sectigo_Public_Server_Authentication_Root_E46.pem
2025-11-23T12:14:26.7825598Z rehash: warning: skipping duplicate certificate in azl_SSL.com_TLS_RSA_Root_CA_2022.pem
2025-11-23T12:14:26.8135072Z rehash: warning: skipping duplicate certificate in azl_SSL.com_TLS_ECC_Root_CA_2022.pem
2025-11-23T12:14:26.9953321Z 4 added, 0 removed; done.
2025-11-23T12:14:26.9953803Z Running hooks in /etc/ca-certificates/update.d...
2025-11-23T12:14:27.0039932Z done.
2025-11-23T12:14:27.0426874Z CA certificates copied and updated successfully.
2025-11-23T12:14:27.2629741Z Site's appCommandLine:
2025-11-23T12:14:27.2645149Z gunicorn main:app --worker-class aiohttp.GunicornWebWorker -w 1 -b 0.0.0.0:$PORT
2025-11-23T12:14:27.264537Z
2025-11-23T12:14:27.26454Z
2025-11-23T12:14:27.2645421Z
2025-11-23T12:14:27.2645439Z
2025-11-23T12:14:27.2645458Z
2025-11-23T12:14:27.2651507Z
2025-11-23T12:14:27.8550544Z Starting periodic command scheduler: cron.
2025-11-23T12:14:27.857131Z Launching oryx with: create-script -appPath /home/site/wwwroot -output /opt/startup/startup.sh -virtualEnvName antenv -defaultApp /opt/defaultsite -userStartupCommand '
2025-11-23T12:14:27.8571576Z gunicorn main:app --worker-class aiohttp.GunicornWebWorker -w 1 -b 0.0.0.0:$PORT
2025-11-23T12:14:27.8571608Z
2025-11-23T12:14:27.8571626Z
2025-11-23T12:14:27.8571644Z
2025-11-23T12:14:27.8571662Z
2025-11-23T12:14:27.857168Z
2025-11-23T12:14:27.8939567Z '
2025-11-23T12:14:28.3575751Z Found build manifest file at '/home/site/wwwroot/oryx-manifest.toml'. Deserializing it...
2025-11-23T12:14:28.4273881Z Output is compressed. Extracting it...
2025-11-23T12:14:28.4274368Z Build Operation ID: 9b1cd7c81d5028ab
2025-11-23T12:14:28.4274419Z Oryx Version: 0.2.20251017.2, Commit: 482d4c55e818733ab33b9d2131f9dc485a21fd03, ReleaseTagName: 20251017.2
2025-11-23T12:14:28.5264852Z Extracting '/home/site/wwwroot/output.tar.gz' to directory '/tmp/8de291f8accb615'...
2025-11-23T12:14:39.3989035Z App path is set to '/tmp/8de291f8accb615'
2025-11-23T12:14:39.418901Z Writing output script to '/opt/startup/startup.sh'
2025-11-23T12:14:39.8019161Z Using packages from virtual environment antenv located at /tmp/8de291f8accb615/antenv.
2025-11-23T12:14:39.80194Z Updated PYTHONPATH to '/opt/startup/app_logs:/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages'
2025-11-23T12:14:45.7745387Z [2025-11-23 12:14:45 +0000] [2111] [INFO] Starting gunicorn 23.0.0
2025-11-23T12:14:45.7923483Z [2025-11-23 12:14:45 +0000] [2111] [INFO] Listening at: http://0.0.0.0:8000 (2111)
2025-11-23T12:14:45.7929512Z [2025-11-23 12:14:45 +0000] [2111] [INFO] Using worker: aiohttp.GunicornWebWorker
2025-11-23T12:14:45.9058944Z [2025-11-23 12:14:45 +0000] [2120] [INFO] Booting worker with pid: 2120
2025-11-23T12:14:48.1094615Z /tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/search/documents/_generated/models/_models_py3.py:993: SyntaxWarning: invalid escape sequence '\ '
2025-11-23T12:14:48.1095266Z   Captions is set to ``extractive``\ , highlighting is enabled by default, and can be configured
2025-11-23T12:14:48.1164403Z /tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/search/documents/_generated/models/_models_py3.py:1184: SyntaxWarning: invalid escape sequence '\ '
2025-11-23T12:14:48.1164641Z   Captions is set to ``extractive``\ , highlighting is enabled by default, and can be configured
2025-11-23T12:14:48.3297444Z /tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/search/documents/_generated/models/_search_index_client_enums.py:84: SyntaxWarning: invalid escape sequence '\ '
2025-11-23T12:14:48.3297824Z   ``extractive``\ , highlighting is enabled by default, and can be configured by appending the
2025-11-23T12:14:48.6917159Z /tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6272: SyntaxWarning: invalid escape sequence '\W'
2025-11-23T12:14:48.6917785Z   pattern: str = "\W+",
2025-11-23T12:14:48.6928398Z /tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6355: SyntaxWarning: invalid escape sequence '\s'
2025-11-23T12:14:48.6928606Z   replace. For example, given the input text "aa bb aa bb", pattern "(aa)\s+(bb)", and
2025-11-23T12:14:48.6943031Z /tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6407: SyntaxWarning: invalid escape sequence '\s'
2025-11-23T12:14:48.6943272Z   replace. For example, given the input text "aa bb aa bb", pattern "(aa)\s+(bb)", and
2025-11-23T12:14:48.7010107Z /tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6497: SyntaxWarning: invalid escape sequence '\W'
2025-11-23T12:14:48.7010348Z   pattern: str = "\W+",
2025-11-23T12:14:49.9672079Z [2025-11-23 12:14:49 +0000] [2120] [ERROR] Exception in worker process
2025-11-23T12:14:49.9672582Z Traceback (most recent call last):
2025-11-23T12:14:49.9672626Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/gunicorn/arbiter.py", line 608, in spawn_worker
2025-11-23T12:14:49.9672649Z     worker.init_process()
2025-11-23T12:14:49.9672674Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/aiohttp/worker.py", line 51, in init_process
2025-11-23T12:14:49.9672694Z     super().init_process()
2025-11-23T12:14:49.9672725Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/gunicorn/workers/base.py", line 135, in init_process
2025-11-23T12:14:49.9672847Z     self.load_wsgi()
2025-11-23T12:14:49.9672873Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/gunicorn/workers/base.py", line 147, in load_wsgi
2025-11-23T12:14:49.9672894Z     self.wsgi = self.app.wsgi()
2025-11-23T12:14:49.9672914Z                 ^^^^^^^^^^^^^^^
2025-11-23T12:14:49.9672939Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/gunicorn/app/base.py", line 66, in wsgi
2025-11-23T12:14:49.9672959Z     self.callable = self.load()
2025-11-23T12:14:49.9672979Z                     ^^^^^^^^^^^
2025-11-23T12:14:49.9673004Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/gunicorn/app/wsgiapp.py", line 57, in load
2025-11-23T12:14:49.9673024Z     return self.load_wsgiapp()
2025-11-23T12:14:49.9673043Z            ^^^^^^^^^^^^^^^^^^^
2025-11-23T12:14:49.9673097Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/gunicorn/app/wsgiapp.py", line 47, in load_wsgiapp
2025-11-23T12:14:49.9673118Z     return util.import_app(self.app_uri)
2025-11-23T12:14:49.967314Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-11-23T12:14:49.9673164Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/gunicorn/util.py", line 370, in import_app
2025-11-23T12:14:49.9673185Z     mod = importlib.import_module(module)
2025-11-23T12:14:49.9673206Z           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-11-23T12:14:49.9673229Z   File "/opt/python/3.12.12/lib/python3.12/importlib/__init__.py", line 90, in import_module
2025-11-23T12:14:49.9673251Z     return _bootstrap._gcd_import(name[level:], package, level)
2025-11-23T12:14:49.9673273Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-11-23T12:14:49.9673296Z   File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
2025-11-23T12:14:49.9673319Z   File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
2025-11-23T12:14:49.9673368Z   File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
2025-11-23T12:14:49.9673391Z   File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
2025-11-23T12:14:49.9673414Z   File "<frozen importlib._bootstrap_external>", line 999, in exec_module
2025-11-23T12:14:49.9673437Z   File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
2025-11-23T12:14:49.9673458Z   File "/tmp/8de291f8accb615/main.py", line 1, in <module>
2025-11-23T12:14:49.9673479Z     from app import create_app
2025-11-23T12:14:49.96735Z   File "/tmp/8de291f8accb615/app.py", line 11, in <module>
2025-11-23T12:14:49.9673524Z     from acs_handler import ACSCallHandler  # NEW: Import ACS handler
2025-11-23T12:14:49.9673545Z     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-11-23T12:14:49.9673567Z   File "/tmp/8de291f8accb615/acs_handler.py", line 4, in <module>
2025-11-23T12:14:49.9673589Z     from azure.communication.callautomation import (
2025-11-23T12:14:49.9673653Z ImportError: cannot import name 'MediaStreamingTransportType' from 'azure.communication.callautomation' (/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/communication/callautomation/__init__.py)
2025-11-23T12:14:49.9822619Z [2025-11-23 12:14:49 +0000] [2120] [INFO] Worker exiting (pid: 2120)
2025-11-23T12:14:50.4470756Z [2025-11-23 12:14:50 +0000] [2111] [ERROR] Worker (pid:2120) exited with code 3
2025-11-23T12:14:50.4483835Z [2025-11-23 12:14:50 +0000] [2111] [ERROR] Shutting down: Master
2025-11-23T12:14:50.4486089Z [2025-11-23 12:14:50 +0000] [2111] [ERROR] Reason: Worker failed to boot.
2025-11-23T12:16:22.896Z No new trace in the past 1 min(s).
2025-11-23T12:17:22.896Z No new trace in the past 2 min(s).
2025-11-23T12:18:22.896Z No new trace in the past 3 min(s).
2025-11-23T12:19:22.896Z No new trace in the past 4 min(s).
