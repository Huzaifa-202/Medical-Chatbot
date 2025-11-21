2025-11-21T13:22:41.7382571Z INFO:voicerag:📩 Received data type: <class 'list'>
2025-11-21T13:22:41.7449408Z INFO:voicerag:📋 Event type: Microsoft.Communication.IncomingCall
2025-11-21T13:22:41.7463774Z INFO:voicerag:📞 Processing incoming call
2025-11-21T13:22:41.7463827Z INFO:voicerag:📞 Caller: +13463755076
2025-11-21T13:22:41.746385Z INFO:voicerag:Getting D365 access token...
2025-11-21T13:22:42.3501866Z INFO:voicerag:✅ D365 token obtained
2025-11-21T13:22:42.4171263Z INFO:voicerag:No customer found for +13463755076
2025-11-21T13:22:42.4232953Z INFO:voicerag:📞 Answering call...
2025-11-21T13:22:42.4414417Z INFO:azure.core.pipeline.policies.http_logging_policy:Request URL: 'https://bizapp-acs.unitedstates.communication.azure.com/calling/callConnections:answer?api-version=REDACTED'
2025-11-21T13:22:42.4414904Z Request method: 'POST'
2025-11-21T13:22:42.4415036Z Request headers:
2025-11-21T13:22:42.4415063Z     'Content-Type': 'application/json'
2025-11-21T13:22:42.4415085Z     'Content-Length': '9072'
2025-11-21T13:22:42.4415108Z     'Repeatability-First-Sent': 'REDACTED'
2025-11-21T13:22:42.441513Z     'Repeatability-Request-ID': 'REDACTED'
2025-11-21T13:22:42.4415152Z     'Accept': 'application/json'
2025-11-21T13:22:42.4415176Z     'x-ms-client-request-id': '28df659c-c6dd-11f0-8db7-42e7a69d7ba0'
2025-11-21T13:22:42.4415204Z     'User-Agent': 'azsdk-python-communication-callautomation/1.5.0 Python/3.12.12 (Linux-6.6.104.2-1.azl3-x86_64-with-glibc2.31)'
2025-11-21T13:22:42.4415226Z     'x-ms-date': 'REDACTED'
2025-11-21T13:22:42.4415247Z     'x-ms-content-sha256': 'REDACTED'
2025-11-21T13:22:42.441527Z     'x-ms-return-client-request-id': 'true'
2025-11-21T13:22:42.4415312Z     'Authorization': 'REDACTED'
2025-11-21T13:22:42.4415334Z A body is sent with the request
2025-11-21T13:22:42.664937Z INFO:azure.core.pipeline.policies.http_logging_policy:Response status: 200
2025-11-21T13:22:42.6649719Z Response headers:
2025-11-21T13:22:42.6649755Z     'Date': 'Fri, 21 Nov 2025 13:22:42 GMT'
2025-11-21T13:22:42.6649788Z     'Content-Type': 'application/json; charset=utf-8'
2025-11-21T13:22:42.6649811Z     'Transfer-Encoding': 'chunked'
2025-11-21T13:22:42.6649834Z     'Connection': 'keep-alive'
2025-11-21T13:22:42.6649855Z     'MS-CV': 'REDACTED'
2025-11-21T13:22:42.6649876Z     'X-Microsoft-Skype-Client': 'REDACTED'
2025-11-21T13:22:42.6649902Z     'X-Ms-Client-Version': 'REDACTED'
2025-11-21T13:22:42.6650025Z     'api-supported-versions': 'REDACTED'
2025-11-21T13:22:42.6650053Z     'x-ms-client-request-id': '28df659c-c6dd-11f0-8db7-42e7a69d7ba0'
2025-11-21T13:22:42.6650075Z     'X-Microsoft-Skype-Chain-ID': 'REDACTED'
2025-11-21T13:22:42.6650097Z     'x-azure-ref': 'REDACTED'
2025-11-21T13:22:42.6650118Z     'Strict-Transport-Security': 'REDACTED'
2025-11-21T13:22:42.665014Z     'X-Cache': 'REDACTED'
2025-11-21T13:22:43.39936Z INFO:voicerag:✅ Call answered: 0d005780-1825-4e56-92d6-fb3bb7cf71b5
2025-11-21T13:22:43.3993845Z INFO:voicerag:🎵 Starting audio streaming to: wss://bizapps-webapp.azurewebsites.net/realtime
2025-11-21T13:22:43.3993887Z ERROR:voicerag:❌ Failed to start audio streaming: CallMediaOperations.start_media_streaming() got multiple values for argument 'start_media_streaming_request'
2025-11-21T13:22:43.3993911Z ERROR:voicerag:   This means customer will hear silence!
2025-11-21T13:22:43.399397Z ERROR:voicerag:Traceback (most recent call last):
2025-11-21T13:22:43.3993995Z   File "/tmp/8de28fb50e1284b/acs_handler.py", line 106, in _handle_call
2025-11-21T13:22:43.3994018Z     call_connection.start_media_streaming(
2025-11-21T13:22:43.3994045Z   File "/tmp/8de28fb50e1284b/antenv/lib/python3.12/site-packages/azure/core/tracing/decorator.py", line 119, in wrapper_use_tracer
2025-11-21T13:22:43.3994067Z     return func(*args, **kwargs)
2025-11-21T13:22:43.3994088Z            ^^^^^^^^^^^^^^^^^^^^^
2025-11-21T13:22:43.3994116Z   File "/tmp/8de28fb50e1284b/antenv/lib/python3.12/site-packages/azure/communication/callautomation/_call_connection_client.py", line 1118, in start_media_streaming
2025-11-21T13:22:43.3994142Z     self._call_media_client.start_media_streaming(self._call_connection_id, start_media_streaming_request, **kwargs)
2025-11-21T13:22:43.3994168Z   File "/tmp/8de28fb50e1284b/antenv/lib/python3.12/site-packages/azure/core/tracing/decorator.py", line 119, in wrapper_use_tracer
2025-11-21T13:22:43.3994189Z     return func(*args, **kwargs)
2025-11-21T13:22:43.3994236Z            ^^^^^^^^^^^^^^^^^^^^^
2025-11-21T13:22:43.3994263Z TypeError: CallMediaOperations.start_media_streaming() got multiple values for argument 'start_media_streaming_request'
2025-11-21T13:22:43.3994284Z
2025-11-21T13:22:43.3999441Z INFO:voicerag:✅ Phone call activity created: 0e59a526-ddc6-f011-bbd3-7c1e527fc4af
2025-11-21T13:22:44.1284395Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallConnected
2025-11-21T13:22:44.1311586Z INFO:voicerag:📨 Callback event: Microsoft.Communication.ParticipantsUpdated
2025-11-21T13:23:53.2293657Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallDisconnected
