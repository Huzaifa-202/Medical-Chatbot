Connected!
2025-11-21T14:55:48.764372Z INFO:voicerag:📩 Received data type: <class 'list'>
2025-11-21T14:55:48.7675791Z INFO:voicerag:📋 Event type: Microsoft.Communication.IncomingCall
2025-11-21T14:55:48.7739579Z INFO:voicerag:📞 Processing incoming call
2025-11-21T14:55:48.7739655Z INFO:voicerag:📞 Caller: +13463755076
2025-11-21T14:55:48.773968Z INFO:voicerag:Getting D365 access token...
2025-11-21T14:55:49.0999286Z INFO:voicerag:✅ D365 token obtained
2025-11-21T14:55:49.2749129Z INFO:voicerag:No customer found for +13463755076
2025-11-21T14:55:49.2756825Z INFO:voicerag:📞 Answering call...
2025-11-21T14:55:49.3298117Z INFO:azure.core.pipeline.policies.http_logging_policy:Request URL: 'https://bizapp-acs.unitedstates.communication.azure.com/calling/callConnections:answer?api-version=REDACTED'
2025-11-21T14:55:49.3298934Z Request method: 'POST'
2025-11-21T14:55:49.329897Z Request headers:
2025-11-21T14:55:49.3298993Z     'Content-Type': 'application/json'
2025-11-21T14:55:49.3299015Z     'Content-Length': '9045'
2025-11-21T14:55:49.3299036Z     'Repeatability-First-Sent': 'REDACTED'
2025-11-21T14:55:49.3299057Z     'Repeatability-Request-ID': 'REDACTED'
2025-11-21T14:55:49.3299079Z     'Accept': 'application/json'
2025-11-21T14:55:49.3299102Z     'x-ms-client-request-id': '2aea1e9c-c6ea-11f0-9bbd-b684df2c9c93'
2025-11-21T14:55:49.3299128Z     'User-Agent': 'azsdk-python-communication-callautomation/1.5.0 Python/3.12.12 (Linux-6.6.104.2-1.azl3-x86_64-with-glibc2.31)'
2025-11-21T14:55:49.3299149Z     'x-ms-date': 'REDACTED'
2025-11-21T14:55:49.329917Z     'x-ms-content-sha256': 'REDACTED'
2025-11-21T14:55:49.3299214Z     'x-ms-return-client-request-id': 'true'
2025-11-21T14:55:49.3299235Z     'Authorization': 'REDACTED'
2025-11-21T14:55:49.3299256Z A body is sent with the request
2025-11-21T14:55:49.8234377Z INFO:azure.core.pipeline.policies.http_logging_policy:Response status: 200
2025-11-21T14:55:49.8235742Z Response headers:
2025-11-21T14:55:49.8235863Z     'Date': 'Fri, 21 Nov 2025 14:55:49 GMT'
2025-11-21T14:55:49.8235895Z     'Content-Type': 'application/json; charset=utf-8'
2025-11-21T14:55:49.8235916Z     'Transfer-Encoding': 'chunked'
2025-11-21T14:55:49.8235938Z     'Connection': 'keep-alive'
2025-11-21T14:55:49.8235959Z     'MS-CV': 'REDACTED'
2025-11-21T14:55:49.823598Z     'X-Microsoft-Skype-Client': 'REDACTED'
2025-11-21T14:55:49.8236157Z     'X-Ms-Client-Version': 'REDACTED'
2025-11-21T14:55:49.8236182Z     'api-supported-versions': 'REDACTED'
2025-11-21T14:55:49.8236206Z     'x-ms-client-request-id': '2aea1e9c-c6ea-11f0-9bbd-b684df2c9c93'
2025-11-21T14:55:49.8236228Z     'X-Microsoft-Skype-Chain-ID': 'REDACTED'
2025-11-21T14:55:49.8236249Z     'x-azure-ref': 'REDACTED'
2025-11-21T14:55:49.8236271Z     'Strict-Transport-Security': 'REDACTED'
2025-11-21T14:55:49.8236293Z     'X-Cache': 'REDACTED'
2025-11-21T14:55:49.8391882Z INFO:voicerag:✅ Call answered: 1c005f80-8116-4bff-8b4a-dc7b8dbcf423
2025-11-21T14:55:49.8432139Z ERROR:voicerag:❌ Failed to start audio streaming: cannot import name 'StartMediaStreamingOptions' from 'azure.communication.callautomation' (/tmp/8de2902905bbf1c/antenv/lib/python3.12/site-packages/azure/communication/callautomation/__init__.py)
2025-11-21T14:55:49.8438881Z ERROR:voicerag:   This means customer will hear silence!
2025-11-21T14:55:49.8598222Z ERROR:voicerag:Traceback (most recent call last):
2025-11-21T14:55:49.8598933Z   File "/tmp/8de2902905bbf1c/acs_handler.py", line 98, in _handle_call
2025-11-21T14:55:49.8598967Z     from azure.communication.callautomation import (
2025-11-21T14:55:49.8598997Z ImportError: cannot import name 'StartMediaStreamingOptions' from 'azure.communication.callautomation' (/tmp/8de2902905bbf1c/antenv/lib/python3.12/site-packages/azure/communication/callautomation/__init__.py)
2025-11-21T14:55:49.8599016Z
2025-11-21T14:55:50.1852112Z INFO:voicerag:✅ Phone call activity created: bb3d3228-eac6-f011-bbd3-7c1e527fc4af
2025-11-21T14:55:51.2000335Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallConnected
2025-11-21T14:55:51.2112793Z INFO:voicerag:📨 Callback event: Microsoft.Communication.ParticipantsUpdated
2025-11-21T14:56:44.4434539Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallDisconnected
