Connected!
2025-11-21T17:03:52.637882Z    _____
2025-11-21T17:03:52.6448687Z   /  _  \ __________ _________   ____
2025-11-21T17:03:52.6448783Z  /  /_\  \\___   /  |  \_  __ \_/ __ \
2025-11-21T17:03:52.6448808Z /    |    \/    /|  |  /|  | \/\  ___/
2025-11-21T17:03:52.6448832Z \____|__  /_____ \____/ |__|    \___  >
2025-11-21T17:03:52.6448863Z         \/      \/                  \/
2025-11-21T17:03:52.6448916Z A P P   S E R V I C E   O N   L I N U X
2025-11-21T17:03:52.6448938Z
2025-11-21T17:03:52.6448969Z Documentation    : http://aka.ms/webapp-linux
2025-11-21T17:03:52.6448994Z Python quickstart: https://aka.ms/python-qs
2025-11-21T17:03:52.6449017Z Python version   : 3.12.12
2025-11-21T17:03:52.6449039Z Instance Name    : lw1sdlwk000853
2025-11-21T17:03:52.6449068Z Instance Id      : 91baf352954053b10aec40a76c08c5d8df4b45053890ea68ab5fc855af1343fa
2025-11-21T17:03:52.6449108Z
2025-11-21T17:03:52.6449132Z Note: Any data outside '/home' is not persisted
2025-11-21T17:03:54.0941756Z Starting OpenBSD Secure Shell server: sshd.
2025-11-21T17:03:54.1221391Z WEBSITES_INCLUDE_CLOUD_CERTS is not set to true.
2025-11-21T17:03:54.1926004Z Updating certificates in /etc/ssl/certs...
2025-11-21T17:04:21.1940286Z rehash: warning: skipping duplicate certificate in azl_Sectigo_Public_Server_Authentication_Root_R46.pem
2025-11-21T17:04:21.2135036Z rehash: warning: skipping duplicate certificate in azl_Sectigo_Public_Server_Authentication_Root_E46.pem
2025-11-21T17:04:21.2602919Z rehash: warning: skipping duplicate certificate in azl_SSL.com_TLS_RSA_Root_CA_2022.pem
2025-11-21T17:04:21.307688Z rehash: warning: skipping duplicate certificate in azl_SSL.com_TLS_ECC_Root_CA_2022.pem
2025-11-21T17:04:21.4452046Z 4 added, 0 removed; done.
2025-11-21T17:04:21.4452641Z Running hooks in /etc/ca-certificates/update.d...
2025-11-21T17:04:21.523266Z done.
2025-11-21T17:04:21.5332912Z CA certificates copied and updated successfully.
2025-11-21T17:04:21.8138059Z Site's appCommandLine:
2025-11-21T17:04:21.8138459Z gunicorn main:app --worker-class aiohttp.GunicornWebWorker -w 1 -b 0.0.0.0:$PORT
2025-11-21T17:04:21.8138513Z
2025-11-21T17:04:21.8138593Z
2025-11-21T17:04:21.813862Z
2025-11-21T17:04:21.8138645Z
2025-11-21T17:04:21.8138671Z
2025-11-21T17:04:21.8138696Z
2025-11-21T17:04:22.8318862Z Starting periodic command scheduler: cron.
2025-11-21T17:04:22.8617287Z Launching oryx with: create-script -appPath /home/site/wwwroot -output /opt/startup/startup.sh -virtualEnvName antenv -defaultApp /opt/defaultsite -userStartupCommand '
2025-11-21T17:04:22.8617721Z gunicorn main:app --worker-class aiohttp.GunicornWebWorker -w 1 -b 0.0.0.0:$PORT
2025-11-21T17:04:22.8617753Z
2025-11-21T17:04:22.8617785Z
2025-11-21T17:04:22.8617806Z
2025-11-21T17:04:22.8617827Z
2025-11-21T17:04:22.8617848Z
2025-11-21T17:04:22.861787Z '
2025-11-21T17:04:24.1154112Z Found build manifest file at '/home/site/wwwroot/oryx-manifest.toml'. Deserializing it...
2025-11-21T17:04:24.2321479Z Build Operation ID: 9b1cd7c81d5028ab
2025-11-21T17:04:24.2322436Z Oryx Version: 0.2.20251017.2, Commit: 482d4c55e818733ab33b9d2131f9dc485a21fd03, ReleaseTagName: 20251017.2
2025-11-21T17:04:24.2774176Z Output is compressed. Extracting it...
2025-11-21T17:04:24.2774629Z Extracting '/home/site/wwwroot/output.tar.gz' to directory '/tmp/8de291f8accb615'...
2025-11-21T17:04:36.5075625Z App path is set to '/tmp/8de291f8accb615'
2025-11-21T17:04:36.5259463Z Writing output script to '/opt/startup/startup.sh'
2025-11-21T17:04:36.978437Z Using packages from virtual environment antenv located at /tmp/8de291f8accb615/antenv.
2025-11-21T17:04:36.9884564Z Updated PYTHONPATH to '/opt/startup/app_logs:/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages'
2025-11-21T17:04:42.0476158Z [2025-11-21 17:04:42 +0000] [2112] [INFO] Starting gunicorn 23.0.0
2025-11-21T17:04:42.0737707Z [2025-11-21 17:04:42 +0000] [2112] [INFO] Listening at: http://0.0.0.0:8000 (2112)
2025-11-21T17:04:42.0751326Z [2025-11-21 17:04:42 +0000] [2112] [INFO] Using worker: aiohttp.GunicornWebWorker
2025-11-21T17:04:42.0923687Z [2025-11-21 17:04:42 +0000] [2120] [INFO] Booting worker with pid: 2120
2025-11-21T17:04:44.0968225Z /tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/search/documents/_generated/models/_models_py3.py:993: SyntaxWarning: invalid escape sequence '\ '
2025-11-21T17:04:44.0968659Z   Captions is set to ``extractive``\ , highlighting is enabled by default, and can be configured
2025-11-21T17:04:44.0978398Z /tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/search/documents/_generated/models/_models_py3.py:1184: SyntaxWarning: invalid escape sequence '\ '
2025-11-21T17:04:44.0978535Z   Captions is set to ``extractive``\ , highlighting is enabled by default, and can be configured
2025-11-21T17:04:44.4803928Z /tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/search/documents/_generated/models/_search_index_client_enums.py:84: SyntaxWarning: invalid escape sequence '\ '
2025-11-21T17:04:44.4804861Z   ``extractive``\ , highlighting is enabled by default, and can be configured by appending the
2025-11-21T17:04:44.8480786Z /tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6272: SyntaxWarning: invalid escape sequence '\W'
2025-11-21T17:04:44.8481379Z   pattern: str = "\W+",
2025-11-21T17:04:44.8490958Z /tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6355: SyntaxWarning: invalid escape sequence '\s'
2025-11-21T17:04:44.8491218Z   replace. For example, given the input text "aa bb aa bb", pattern "(aa)\s+(bb)", and
2025-11-21T17:04:44.8756657Z /tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6407: SyntaxWarning: invalid escape sequence '\s'
2025-11-21T17:04:44.8756998Z   replace. For example, given the input text "aa bb aa bb", pattern "(aa)\s+(bb)", and
2025-11-21T17:04:44.8757046Z /tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6497: SyntaxWarning: invalid escape sequence '\W'
2025-11-21T17:04:44.8757069Z   pattern: str = "\W+",
2025-11-21T17:04:46.3315128Z [2025-11-21 17:04:46 +0000] [2120] [ERROR] Exception in worker process
2025-11-21T17:04:46.3315776Z Traceback (most recent call last):
2025-11-21T17:04:46.3315826Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/gunicorn/arbiter.py", line 608, in spawn_worker
2025-11-21T17:04:46.3315866Z     worker.init_process()
2025-11-21T17:04:46.3315901Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/aiohttp/worker.py", line 51, in init_process
2025-11-21T17:04:46.3315929Z     super().init_process()
2025-11-21T17:04:46.331605Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/gunicorn/workers/base.py", line 135, in init_process
2025-11-21T17:04:46.3316079Z     self.load_wsgi()
2025-11-21T17:04:46.331611Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/gunicorn/workers/base.py", line 147, in load_wsgi
2025-11-21T17:04:46.3316139Z     self.wsgi = self.app.wsgi()
2025-11-21T17:04:46.3316167Z                 ^^^^^^^^^^^^^^^
2025-11-21T17:04:46.3316196Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/gunicorn/app/base.py", line 66, in wsgi
2025-11-21T17:04:46.3316245Z     self.callable = self.load()
2025-11-21T17:04:46.3316448Z                     ^^^^^^^^^^^
2025-11-21T17:04:46.3316478Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/gunicorn/app/wsgiapp.py", line 57, in load
2025-11-21T17:04:46.3316506Z     return self.load_wsgiapp()
2025-11-21T17:04:46.3316533Z            ^^^^^^^^^^^^^^^^^^^
2025-11-21T17:04:46.3316599Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/gunicorn/app/wsgiapp.py", line 47, in load_wsgiapp
2025-11-21T17:04:46.3316652Z     return util.import_app(self.app_uri)
2025-11-21T17:04:46.3316682Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-11-21T17:04:46.3316712Z   File "/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/gunicorn/util.py", line 370, in import_app
2025-11-21T17:04:46.3316741Z     mod = importlib.import_module(module)
2025-11-21T17:04:46.3316769Z           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-11-21T17:04:46.33168Z   File "/opt/python/3.12.12/lib/python3.12/importlib/__init__.py", line 90, in import_module
2025-11-21T17:04:46.3316853Z     return _bootstrap._gcd_import(name[level:], package, level)
2025-11-21T17:04:46.3316884Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-11-21T17:04:46.3316914Z   File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
2025-11-21T17:04:46.3316944Z   File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
2025-11-21T17:04:46.3316974Z   File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
2025-11-21T17:04:46.3317003Z   File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
2025-11-21T17:04:46.3317055Z   File "<frozen importlib._bootstrap_external>", line 999, in exec_module
2025-11-21T17:04:46.3317085Z   File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
2025-11-21T17:04:46.3317228Z   File "/tmp/8de291f8accb615/main.py", line 1, in <module>
2025-11-21T17:04:46.331727Z     from app import create_app
2025-11-21T17:04:46.33173Z   File "/tmp/8de291f8accb615/app.py", line 11, in <module>
2025-11-21T17:04:46.3317349Z     from acs_handler import ACSCallHandler  # NEW: Import ACS handler
2025-11-21T17:04:46.3317378Z     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-11-21T17:04:46.3317406Z   File "/tmp/8de291f8accb615/acs_handler.py", line 4, in <module>
2025-11-21T17:04:46.3317435Z     from azure.communication.callautomation import (
2025-11-21T17:04:46.3317473Z ImportError: cannot import name 'MediaStreamingTransportType' from 'azure.communication.callautomation' (/tmp/8de291f8accb615/antenv/lib/python3.12/site-packages/azure/communication/callautomation/__init__.py)
2025-11-21T17:04:46.346361Z [2025-11-21 17:04:46 +0000] [2120] [INFO] Worker exiting (pid: 2120)
2025-11-21T17:04:46.8145831Z [2025-11-21 17:04:46 +0000] [2112] [ERROR] Worker (pid:2120) exited with code 3
2025-11-21T17:04:46.8244687Z [2025-11-21 17:04:46 +0000] [2112] [ERROR] Shutting down: Master
2025-11-21T17:04:46.8257621Z [2025-11-21 17:04:46 +0000] [2112] [ERROR] Reason: Worker failed to boot.
2025-11-21T17:06:10.736Z No new trace in the past 1 min(s).
2025-11-21T17:07:10.736Z No new trace in the past 2 min(s).
