Connected!
2025-11-21T07:33:37.0357298Z    _____
2025-11-21T07:33:37.0543468Z   /  _  \ __________ _________   ____
2025-11-21T07:33:37.0625861Z  /  /_\  \\___   /  |  \_  __ \_/ __ \
2025-11-21T07:33:37.070356Z /    |    \/    /|  |  /|  | \/\  ___/
2025-11-21T07:33:37.0719458Z \____|__  /_____ \____/ |__|    \___  >
2025-11-21T07:33:37.0801652Z         \/      \/                  \/
2025-11-21T07:33:37.0820363Z A P P   S E R V I C E   O N   L I N U X
2025-11-21T07:33:37.0820427Z
2025-11-21T07:33:37.0820463Z Documentation    : http://aka.ms/webapp-linux
2025-11-21T07:33:37.0820491Z Python quickstart: https://aka.ms/python-qs
2025-11-21T07:33:37.0820517Z Python version   : 3.12.12
2025-11-21T07:33:37.082056Z Instance Name    : lw1sdlwk000853
2025-11-21T07:33:37.0820602Z Instance Id      : 91baf352954053b10aec40a76c08c5d8df4b45053890ea68ab5fc855af1343fa
2025-11-21T07:33:37.0820648Z
2025-11-21T07:33:37.0820676Z Note: Any data outside '/home' is not persisted
2025-11-21T07:33:39.0268592Z Starting OpenBSD Secure Shell server: sshd.
2025-11-21T07:33:39.0556871Z WEBSITES_INCLUDE_CLOUD_CERTS is not set to true.
2025-11-21T07:33:39.1325964Z Updating certificates in /etc/ssl/certs...
2025-11-21T07:34:00.2590786Z rehash: warning: skipping duplicate certificate in azl_Sectigo_Public_Server_Authentication_Root_R46.pem
2025-11-21T07:34:00.3352363Z rehash: warning: skipping duplicate certificate in azl_Sectigo_Public_Server_Authentication_Root_E46.pem
2025-11-21T07:34:00.3352949Z rehash: warning: skipping duplicate certificate in azl_SSL.com_TLS_RSA_Root_CA_2022.pem
2025-11-21T07:34:00.4512869Z rehash: warning: skipping duplicate certificate in azl_SSL.com_TLS_ECC_Root_CA_2022.pem
2025-11-21T07:34:00.6104784Z 4 added, 0 removed; done.
2025-11-21T07:34:00.639039Z Running hooks in /etc/ca-certificates/update.d...
2025-11-21T07:34:00.6467856Z done.
2025-11-21T07:34:00.6960637Z CA certificates copied and updated successfully.
2025-11-21T07:34:01.1910704Z Site's appCommandLine:
2025-11-21T07:34:01.1911613Z gunicorn main:app --worker-class aiohttp.GunicornWebWorker -w 1 -b 0.0.0.0:$PORT
2025-11-21T07:34:01.1911854Z
2025-11-21T07:34:01.1911883Z
2025-11-21T07:34:01.1911906Z
2025-11-21T07:34:01.1911935Z
2025-11-21T07:34:01.191196Z
2025-11-21T07:34:01.1911984Z
2025-11-21T07:34:01.9916902Z Starting periodic command scheduler: cron.
2025-11-21T07:34:01.9917321Z Launching oryx with: create-script -appPath /home/site/wwwroot -output /opt/startup/startup.sh -virtualEnvName antenv -defaultApp /opt/defaultsite -userStartupCommand '
2025-11-21T07:34:01.9917372Z gunicorn main:app --worker-class aiohttp.GunicornWebWorker -w 1 -b 0.0.0.0:$PORT
2025-11-21T07:34:01.9917395Z
2025-11-21T07:34:01.9917421Z
2025-11-21T07:34:01.9917442Z
2025-11-21T07:34:01.9917527Z
2025-11-21T07:34:01.9917549Z
2025-11-21T07:34:01.9917569Z '
2025-11-21T07:34:02.6224324Z Found build manifest file at '/home/site/wwwroot/oryx-manifest.toml'. Deserializing it...
2025-11-21T07:34:02.6495468Z Build Operation ID: ac3d571fcdb8524a
2025-11-21T07:34:02.6597003Z Oryx Version: 0.2.20251017.2, Commit: 482d4c55e818733ab33b9d2131f9dc485a21fd03, ReleaseTagName: 20251017.2
2025-11-21T07:34:02.6597884Z Output is compressed. Extracting it...
2025-11-21T07:34:02.6879068Z Extracting '/home/site/wwwroot/output.tar.gz' to directory '/tmp/8de28cffb3d0ff6'...
2025-11-21T07:34:19.7564284Z App path is set to '/tmp/8de28cffb3d0ff6'
2025-11-21T07:34:19.7856232Z Writing output script to '/opt/startup/startup.sh'
2025-11-21T07:34:20.4391795Z Using packages from virtual environment antenv located at /tmp/8de28cffb3d0ff6/antenv.
2025-11-21T07:34:20.4573389Z Updated PYTHONPATH to '/opt/startup/app_logs:/tmp/8de28cffb3d0ff6/antenv/lib/python3.12/site-packages'
2025-11-21T07:34:27.3467154Z [2025-11-21 07:34:27 +0000] [2112] [INFO] Starting gunicorn 23.0.0
2025-11-21T07:34:27.3550837Z [2025-11-21 07:34:27 +0000] [2112] [INFO] Listening at: http://0.0.0.0:8000 (2112)
2025-11-21T07:34:27.3551369Z [2025-11-21 07:34:27 +0000] [2112] [INFO] Using worker: aiohttp.GunicornWebWorker
2025-11-21T07:34:27.3688761Z [2025-11-21 07:34:27 +0000] [2120] [INFO] Booting worker with pid: 2120
2025-11-21T07:34:29.1311574Z /tmp/8de28cffb3d0ff6/antenv/lib/python3.12/site-packages/azure/search/documents/_generated/models/_models_py3.py:993: SyntaxWarning: invalid escape sequence '\ '
2025-11-21T07:34:29.1312284Z   Captions is set to ``extractive``\ , highlighting is enabled by default, and can be configured
2025-11-21T07:34:29.1384776Z /tmp/8de28cffb3d0ff6/antenv/lib/python3.12/site-packages/azure/search/documents/_generated/models/_models_py3.py:1184: SyntaxWarning: invalid escape sequence '\ '
2025-11-21T07:34:29.1385053Z   Captions is set to ``extractive``\ , highlighting is enabled by default, and can be configured
2025-11-21T07:34:29.3055144Z /tmp/8de28cffb3d0ff6/antenv/lib/python3.12/site-packages/azure/search/documents/_generated/models/_search_index_client_enums.py:84: SyntaxWarning: invalid escape sequence '\ '
2025-11-21T07:34:29.3055791Z   ``extractive``\ , highlighting is enabled by default, and can be configured by appending the
2025-11-21T07:34:29.7644565Z /tmp/8de28cffb3d0ff6/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6272: SyntaxWarning: invalid escape sequence '\W'
2025-11-21T07:34:29.764515Z   pattern: str = "\W+",
2025-11-21T07:34:29.8076751Z /tmp/8de28cffb3d0ff6/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6355: SyntaxWarning: invalid escape sequence '\s'
2025-11-21T07:34:29.8077201Z   replace. For example, given the input text "aa bb aa bb", pattern "(aa)\s+(bb)", and
2025-11-21T07:34:29.8077281Z /tmp/8de28cffb3d0ff6/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6407: SyntaxWarning: invalid escape sequence '\s'
2025-11-21T07:34:29.8077308Z   replace. For example, given the input text "aa bb aa bb", pattern "(aa)\s+(bb)", and
2025-11-21T07:34:29.8077338Z /tmp/8de28cffb3d0ff6/antenv/lib/python3.12/site-packages/azure/search/documents/indexes/_generated/models/_models_py3.py:6497: SyntaxWarning: invalid escape sequence '\W'
2025-11-21T07:34:29.8077529Z   pattern: str = "\W+",
2025-11-21T07:34:31.2284764Z INFO:voicerag:Running in development mode, loading from .env file
2025-11-21T07:34:31.2515912Z INFO:voicerag:✅ D365 integration enabled
2025-11-21T07:34:31.2534366Z INFO:voicerag:Realtime voice choice set to shimmer
2025-11-21T07:34:32.1385126Z INFO:voicerag:✅ ACS call handling enabled
2025-11-21T07:34:32.1397007Z INFO:voicerag:   Incoming calls: https://bizapps-webapp.azurewebsites.net/api/incomingCall
2025-11-21T07:34:32.1415371Z INFO:voicerag:   Callbacks: https://bizapps-webapp.azurewebsites.net/api/callbacks
2025-11-21T07:41:27.4587506Z INFO:voicerag:📩 Received data type: <class 'list'>
2025-11-21T07:41:27.4589083Z INFO:voicerag:📋 Event type: Microsoft.EventGrid.SubscriptionValidationEvent
2025-11-21T07:41:27.4589128Z INFO:voicerag:✅ Event Grid validation - responding
2025-11-21T07:41:34.5032528Z INFO:voicerag:📩 Received data type: <class 'dict'>
2025-11-21T07:41:34.5033386Z INFO:voicerag:📞 Processing incoming call
2025-11-21T07:41:34.503342Z INFO:voicerag:Call data keys: dict_keys(['name'])
2025-11-21T07:41:34.5119448Z ERROR:voicerag:❌ No incomingCallContext. Data: {'name': 'Add your name in the body'}
2025-11-21T07:41:42.4448322Z INFO:voicerag:📩 Received data type: <class 'dict'>
2025-11-21T07:41:42.4454344Z INFO:voicerag:📞 Processing incoming call
2025-11-21T07:41:42.445797Z INFO:voicerag:Call data keys: dict_keys(['name'])
2025-11-21T07:41:42.4532997Z ERROR:voicerag:❌ No incomingCallContext. Data: {'name': 'Add your name in the body'}
2025-11-21T07:59:06.281Z No new trace in the past 1 min(s).
2025-11-21T08:00:06.281Z No new trace in the past 2 min(s).
2025-11-21T08:01:06.281Z No new trace in the past 3 min(s).
2025-11-21T08:02:06.281Z No new trace in the past 4 min(s).
2025-11-21T08:03:06.281Z No new trace in the past 5 min(s).
2025-11-21T08:04:06.281Z No new trace in the past 6 min(s).
