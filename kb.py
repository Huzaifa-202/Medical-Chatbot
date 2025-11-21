Connected!
2025-11-21T15:17:13.5192565Z    _____
2025-11-21T15:17:13.6644718Z   /  _  \ __________ _________   ____
2025-11-21T15:17:13.6645023Z  /  /_\  \\___   /  |  \_  __ \_/ __ \
2025-11-21T15:17:13.6645061Z /    |    \/    /|  |  /|  | \/\  ___/
2025-11-21T15:17:13.6645088Z \____|__  /_____ \____/ |__|    \___  >
2025-11-21T15:17:13.6645116Z         \/      \/                  \/
2025-11-21T15:17:13.664517Z A P P   S E R V I C E   O N   L I N U X
2025-11-21T15:17:13.6645195Z
2025-11-21T15:17:13.664523Z Documentation    : http://aka.ms/webapp-linux
2025-11-21T15:17:13.6645281Z Python quickstart: https://aka.ms/python-qs
2025-11-21T15:17:13.6645319Z Python version   : 3.12.12
2025-11-21T15:17:13.6645346Z Instance Name    : lw1sdlwk000853
2025-11-21T15:17:13.6645384Z Instance Id      : 91baf352954053b10aec40a76c08c5d8df4b45053890ea68ab5fc855af1343fa
2025-11-21T15:17:13.664541Z
2025-11-21T15:17:13.6645437Z Note: Any data outside '/home' is not persisted
2025-11-21T15:17:18.4763524Z Starting OpenBSD Secure Shell server: sshd.
2025-11-21T15:17:18.5067607Z WEBSITES_INCLUDE_CLOUD_CERTS is not set to true.
2025-11-21T15:17:18.6844344Z Updating certificates in /etc/ssl/certs...
2025-11-21T15:18:06.2951739Z rehash: warning: skipping duplicate certificate in azl_Sectigo_Public_Server_Authentication_Root_R46.pem
2025-11-21T15:18:06.362828Z rehash: warning: skipping duplicate certificate in azl_Sectigo_Public_Server_Authentication_Root_E46.pem
2025-11-21T15:18:06.3628944Z rehash: warning: skipping duplicate certificate in azl_SSL.com_TLS_RSA_Root_CA_2022.pem
2025-11-21T15:18:06.4793538Z rehash: warning: skipping duplicate certificate in azl_SSL.com_TLS_ECC_Root_CA_2022.pem
2025-11-21T15:18:06.6831563Z 4 added, 0 removed; done.
2025-11-21T15:18:06.6832013Z Running hooks in /etc/ca-certificates/update.d...
2025-11-21T15:18:06.6926898Z done.
2025-11-21T15:18:06.7501216Z CA certificates copied and updated successfully.
2025-11-21T15:18:06.9826039Z Site's appCommandLine:
2025-11-21T15:18:06.9826845Z gunicorn main:app --worker-class aiohttp.GunicornWebWorker -w 1 -b 0.0.0.0:$PORT
2025-11-21T15:18:06.9826916Z
2025-11-21T15:18:06.982704Z
2025-11-21T15:18:06.9827062Z
2025-11-21T15:18:06.9827084Z
2025-11-21T15:18:06.9827105Z
2025-11-21T15:18:06.9827126Z
2025-11-21T15:18:08.2743665Z Starting periodic command scheduler: cron.
2025-11-21T15:18:08.2744335Z Launching oryx with: create-script -appPath /home/site/wwwroot -output /opt/startup/startup.sh -virtualEnvName antenv -defaultApp /opt/defaultsite -userStartupCommand '
2025-11-21T15:18:08.2744474Z gunicorn main:app --worker-class aiohttp.GunicornWebWorker -w 1 -b 0.0.0.0:$PORT
2025-11-21T15:18:08.2744503Z
2025-11-21T15:18:08.2744534Z
2025-11-21T15:18:08.2744559Z
2025-11-21T15:18:08.2744582Z
2025-11-21T15:18:08.2744605Z
2025-11-21T15:18:08.2744631Z '
2025-11-21T15:18:09.2245698Z Found build manifest file at '/home/site/wwwroot/oryx-manifest.toml'. Deserializing it...
2025-11-21T15:18:09.2702433Z Build Operation ID: a9c95e7499c93a7e
2025-11-21T15:18:09.28911Z Output is compressed. Extracting it...
2025-11-21T15:18:09.2892199Z Oryx Version: 0.2.20251017.2, Commit: 482d4c55e818733ab33b9d2131f9dc485a21fd03, ReleaseTagName: 20251017.2
2025-11-21T15:18:09.3197402Z Extracting '/home/site/wwwroot/output.tar.gz' to directory '/tmp/8de2910b1b91986'...
2025-11-21T15:18:25.5331223Z App path is set to '/tmp/8de2910b1b91986'
2025-11-21T15:18:25.545494Z Writing output script to '/opt/startup/startup.sh'
2025-11-21T15:18:26.2213539Z Using packages from virtual environment antenv located at /tmp/8de2910b1b91986/antenv.
2025-11-21T15:18:26.221459Z Updated PYTHONPATH to '/opt/startup/app_logs:/tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages'
2025-11-21T15:18:37.837794Z [2025-11-21 15:18:37 +0000] [2112] [INFO] Starting gunicorn 23.0.0
2025-11-21T15:18:37.8395932Z [2025-11-21 15:18:37 +0000] [2112] [INFO] Listening at: http://0.0.0.0:8000 (2112)
2025-11-21T15:18:37.840695Z [2025-11-21 15:18:37 +0000] [2112] [INFO] Using worker: aiohttp.GunicornWebWorker
2025-11-21T15:18:37.8776923Z [2025-11-21 15:18:37 +0000] [2120] [INFO] Booting worker with pid: 2120
2025-11-21T15:18:43.4104411Z /tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/azure/search/documents/_generated/models/_models_py3.py:993: SyntaxWarning: invalid escape sequence '\ '
2025-11-21T15:18:43.4104941Z   Captions is set to ``extractive``\ , highlighting is enabled by default, and can be configured
2025-11-21T15:18:43.4135025Z /tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/azure/search/documents/_generated/models/_models_py3.py:1184: SyntaxWarning: invalid escape sequence '\ '
2025-11-21T15:18:43.4135275Z   Captions is set to ``extractive``\ , highlighting is enabled by default, and can be configured
2025-11-21T15:18:43.9261671Z /tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/azure/search/documents/_generated/models/_search_index_client_enums.py:84: SyntaxWarning: invalid escape sequence '\ '
2025-11-21T15:18:43.9262647Z   ``extractive``\ , highlighting is enabled by default, and can be configured by appending the
2025-11-21T15:18:44.7580981Z /tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6272: SyntaxWarning: invalid escape sequence '\W'
2025-11-21T15:18:44.758162Z   pattern: str = "\W+",
2025-11-21T15:18:44.7581672Z /tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6355: SyntaxWarning: invalid escape sequence '\s'
2025-11-21T15:18:44.7581706Z   replace. For example, given the input text "aa bb aa bb", pattern "(aa)\s+(bb)", and
2025-11-21T15:18:44.7581824Z /tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6407: SyntaxWarning: invalid escape sequence '\s'
2025-11-21T15:18:44.7581853Z   replace. For example, given the input text "aa bb aa bb", pattern "(aa)\s+(bb)", and
2025-11-21T15:18:44.7581883Z /tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6497: SyntaxWarning: invalid escape sequence '\W'
2025-11-21T15:18:44.7581907Z   pattern: str = "\W+",
2025-11-21T15:18:47.8437555Z [2025-11-21 15:18:47 +0000] [2120] [ERROR] Exception in worker process
2025-11-21T15:18:47.8438188Z Traceback (most recent call last):
2025-11-21T15:18:47.8438233Z   File "/tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/gunicorn/arbiter.py", line 608, in spawn_worker
2025-11-21T15:18:47.8438258Z     worker.init_process()
2025-11-21T15:18:47.8438291Z   File "/tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/aiohttp/worker.py", line 51, in init_process
2025-11-21T15:18:47.8438319Z     super().init_process()
2025-11-21T15:18:47.8438347Z   File "/tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/gunicorn/workers/base.py", line 135, in init_process
2025-11-21T15:18:47.8438392Z     self.load_wsgi()
2025-11-21T15:18:47.843842Z   File "/tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/gunicorn/workers/base.py", line 147, in load_wsgi
2025-11-21T15:18:47.8438446Z     self.wsgi = self.app.wsgi()
2025-11-21T15:18:47.8438469Z                 ^^^^^^^^^^^^^^^
2025-11-21T15:18:47.8438496Z   File "/tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/gunicorn/app/base.py", line 66, in wsgi
2025-11-21T15:18:47.843852Z     self.callable = self.load()
2025-11-21T15:18:47.8438565Z                     ^^^^^^^^^^^
2025-11-21T15:18:47.8438593Z   File "/tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/gunicorn/app/wsgiapp.py", line 57, in load
2025-11-21T15:18:47.8438616Z     return self.load_wsgiapp()
2025-11-21T15:18:47.8438639Z            ^^^^^^^^^^^^^^^^^^^
2025-11-21T15:18:47.8438685Z   File "/tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/gunicorn/app/wsgiapp.py", line 47, in load_wsgiapp
2025-11-21T15:18:47.843871Z     return util.import_app(self.app_uri)
2025-11-21T15:18:47.8438753Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-11-21T15:18:47.843878Z   File "/tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/gunicorn/util.py", line 370, in import_app
2025-11-21T15:18:47.8438806Z     mod = importlib.import_module(module)
2025-11-21T15:18:47.843883Z           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-11-21T15:18:47.8438856Z   File "/opt/python/3.12.12/lib/python3.12/importlib/__init__.py", line 90, in import_module
2025-11-21T15:18:47.8438882Z     return _bootstrap._gcd_import(name[level:], package, level)
2025-11-21T15:18:47.8438926Z            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-11-21T15:18:47.8438951Z   File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
2025-11-21T15:18:47.8438977Z   File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
2025-11-21T15:18:47.8439002Z   File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
2025-11-21T15:18:47.8439028Z   File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
2025-11-21T15:18:47.8439053Z   File "<frozen importlib._bootstrap_external>", line 999, in exec_module
2025-11-21T15:18:47.8439103Z   File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
2025-11-21T15:18:47.8524908Z   File "/tmp/8de2910b1b91986/main.py", line 1, in <module>
2025-11-21T15:18:47.8525066Z     from app import create_app
2025-11-21T15:18:47.8525098Z   File "/tmp/8de2910b1b91986/app.py", line 11, in <module>
2025-11-21T15:18:47.8525125Z     from acs_handler import ACSCallHandler  # NEW: Import ACS handler
2025-11-21T15:18:47.852515Z     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2025-11-21T15:18:47.8525411Z   File "/tmp/8de2910b1b91986/acs_handler.py", line 4, in <module>
2025-11-21T15:18:47.852544Z     from azure.communication.callautomation import (
2025-11-21T15:18:47.8525635Z ImportError: cannot import name 'MediaStreamingTransportType' from 'azure.communication.callautomation' (/tmp/8de2910b1b91986/antenv/lib/python3.12/site-packages/azure/communication/callautomation/__init__.py)
2025-11-21T15:18:47.8525905Z [2025-11-21 15:18:47 +0000] [2120] [INFO] Worker exiting (pid: 2120)
2025-11-21T15:18:48.9795499Z [2025-11-21 15:18:48 +0000] [2112] [ERROR] Worker (pid:2120) exited with code 3
2025-11-21T15:18:48.9973986Z [2025-11-21 15:18:48 +0000] [2112] [ERROR] Shutting down: Master
2025-11-21T15:18:48.9974377Z [2025-11-21 15:18:48 +0000] [2112] [ERROR] Reason: Worker failed to boot.
