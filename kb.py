025-11-21T16:49:48.920581Z INFO:voicerag:📩 Received data type: <class 'list'>
2025-11-21T16:49:48.9311734Z INFO:voicerag:📋 Event type: Microsoft.Communication.IncomingCall
2025-11-21T16:49:49.7373121Z INFO:voicerag:📞 Processing incoming call
2025-11-21T16:49:49.7373388Z INFO:voicerag:📞 Caller: +13463755076
2025-11-21T16:49:49.7373414Z INFO:voicerag:Getting D365 access token...
2025-11-21T16:49:49.7373437Z INFO:voicerag:✅ D365 token obtained
2025-11-21T16:49:49.7373457Z INFO:voicerag:No customer found for +13463755076
2025-11-21T16:49:49.7377296Z ERROR:voicerag:⚠️ Media streaming classes not available: cannot import name 'MediaStreamingTransportType' from 'azure.communication.callautomation._models' (/tmp/8de2913af9750e5/antenv/lib/python3.12/site-packages/azure/communication/callautomation/_models.py)
2025-11-21T16:49:49.7378014Z INFO:voicerag:   Call will be answered without streaming
2025-11-21T16:49:49.7378069Z INFO:voicerag:📞 Answering call...
2025-11-21T16:49:49.7378096Z INFO:azure.core.pipeline.policies.http_logging_policy:Request URL: 'https://bizapp-acs.unitedstates.communication.azure.com/calling/callConnections:answer?api-version=REDACTED'
2025-11-21T16:49:49.7378115Z Request method: 'POST'
2025-11-21T16:49:49.7378172Z Request headers:
2025-11-21T16:49:49.7378192Z     'Content-Type': 'application/json'
2025-11-21T16:49:49.7378211Z     'Content-Length': '9069'
2025-11-21T16:49:49.737823Z     'Repeatability-First-Sent': 'REDACTED'
2025-11-21T16:49:49.7378249Z     'Repeatability-Request-ID': 'REDACTED'
2025-11-21T16:49:49.7378268Z     'Accept': 'application/json'
2025-11-21T16:49:49.7378288Z     'x-ms-client-request-id': '17fd6e50-c6fa-11f0-8419-deedb78fa783'
2025-11-21T16:49:49.7378464Z     'User-Agent': 'azsdk-python-communication-callautomation/1.5.0 Python/3.12.12 (Linux-6.6.104.2-1.azl3-x86_64-with-glibc2.31)'
2025-11-21T16:49:49.7378495Z     'x-ms-date': 'REDACTED'
2025-11-21T16:49:49.7378515Z     'x-ms-content-sha256': 'REDACTED'
2025-11-21T16:49:49.7378535Z     'x-ms-return-client-request-id': 'true'
2025-11-21T16:49:49.7378575Z     'Authorization': 'REDACTED'
2025-11-21T16:49:49.7378595Z A body is sent with the request
2025-11-21T16:49:50.2692325Z INFO:azure.core.pipeline.policies.http_logging_policy:Response status: 200
2025-11-21T16:49:50.2692933Z Response headers:
2025-11-21T16:49:50.2692971Z     'Date': 'Fri, 21 Nov 2025 16:49:50 GMT'
2025-11-21T16:49:50.2692999Z     'Content-Type': 'application/json; charset=utf-8'
2025-11-21T16:49:50.2693021Z     'Transfer-Encoding': 'chunked'
2025-11-21T16:49:50.2693045Z     'Connection': 'keep-alive'
2025-11-21T16:49:50.2693071Z     'MS-CV': 'REDACTED'
2025-11-21T16:49:50.2693093Z     'X-Microsoft-Skype-Client': 'REDACTED'
2025-11-21T16:49:50.2693114Z     'X-Ms-Client-Version': 'REDACTED'
2025-11-21T16:49:50.2693231Z     'api-supported-versions': 'REDACTED'
2025-11-21T16:49:50.269583Z     'x-ms-client-request-id': '17fd6e50-c6fa-11f0-8419-deedb78fa783'
2025-11-21T16:49:50.2695885Z     'X-Microsoft-Skype-Chain-ID': 'REDACTED'
2025-11-21T16:49:50.2695905Z     'x-azure-ref': 'REDACTED'
2025-11-21T16:49:50.2695925Z     'Strict-Transport-Security': 'REDACTED'
2025-11-21T16:49:50.2695945Z     'X-Cache': 'REDACTED'
2025-11-21T16:49:50.2845536Z INFO:voicerag:✅ Call answered (no streaming): 23005c80-c7b9-49a5-9095-9031c3d0ffa7
2025-11-21T16:49:51.1506391Z INFO:voicerag:✅ Phone call activity created: 643f9713-fac6-f011-bbd3-7c1e527fc4af
2025-11-21T16:49:51.1518415Z INFO:voicerag:📨 Callback event: Microsoft.Communication.ParticipantsUpdated
2025-11-21T16:49:51.1576886Z INFO:voicerag:✅ Phone call activity created: 643f9713-fac6-f011-bbd3-7c1e527fc4af
2025-11-21T16:49:51.1586937Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallConnected
2025-11-21T16:50:55.1289858Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallDisconnected
