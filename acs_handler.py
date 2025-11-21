Connected!
2025-11-21T10:35:55.8348202Z INFO:voicerag:📩 Received data type: <class 'list'>
2025-11-21T10:35:55.8376472Z INFO:voicerag:📋 Event type: Microsoft.Communication.IncomingCall
2025-11-21T10:35:55.8380596Z INFO:voicerag:📞 Processing incoming call
2025-11-21T10:35:55.8380648Z INFO:voicerag:Call data keys: dict_keys(['to', 'from', 'serverCallId', 'callerDisplayName', 'incomingCallContext', 'correlationId'])
2025-11-21T10:35:55.8380666Z INFO:voicerag:📞 Caller: +13463755076
2025-11-21T10:35:55.8380684Z INFO:voicerag:Getting D365 access token...
2025-11-21T10:35:56.4449579Z INFO:voicerag:✅ D365 token obtained
2025-11-21T10:35:56.7322648Z INFO:voicerag:No customer found for +13463755076
2025-11-21T10:35:56.7342323Z INFO:voicerag:📞 Answering call...
2025-11-21T10:35:56.7378636Z INFO:azure.core.pipeline.policies.http_logging_policy:Request URL: 'https://bizapp-acs.unitedstates.communication.azure.com/calling/callConnections:answer?api-version=REDACTED'
2025-11-21T10:35:56.7378827Z Request method: 'POST'
2025-11-21T10:35:56.7378851Z Request headers:
2025-11-21T10:35:56.737887Z     'Content-Type': 'application/json'
2025-11-21T10:35:56.7378889Z     'Content-Length': '9072'
2025-11-21T10:35:56.7378908Z     'Repeatability-First-Sent': 'REDACTED'
2025-11-21T10:35:56.7378927Z     'Repeatability-Request-ID': 'REDACTED'
2025-11-21T10:35:56.7378945Z     'Accept': 'application/json'
2025-11-21T10:35:56.7378965Z     'x-ms-client-request-id': 'dd03ce04-c6c5-11f0-b822-46a038d79753'
2025-11-21T10:35:56.7378988Z     'User-Agent': 'azsdk-python-communication-callautomation/1.5.0 Python/3.12.12 (Linux-6.6.104.2-1.azl3-x86_64-with-glibc2.31)'
2025-11-21T10:35:56.7379007Z     'x-ms-date': 'REDACTED'
2025-11-21T10:35:56.7379061Z     'x-ms-content-sha256': 'REDACTED'
2025-11-21T10:35:56.737908Z     'x-ms-return-client-request-id': 'true'
2025-11-21T10:35:56.7379207Z     'Authorization': 'REDACTED'
2025-11-21T10:35:56.737923Z A body is sent with the request
2025-11-21T10:35:57.1736138Z INFO:azure.core.pipeline.policies.http_logging_policy:Response status: 200
2025-11-21T10:35:57.1737151Z Response headers:
2025-11-21T10:35:57.1737188Z     'Date': 'Fri, 21 Nov 2025 10:35:57 GMT'
2025-11-21T10:35:57.173721Z     'Content-Type': 'application/json; charset=utf-8'
2025-11-21T10:35:57.1737229Z     'Transfer-Encoding': 'chunked'
2025-11-21T10:35:57.1737247Z     'Connection': 'keep-alive'
2025-11-21T10:35:57.1737349Z     'MS-CV': 'REDACTED'
2025-11-21T10:35:57.173737Z     'X-Microsoft-Skype-Client': 'REDACTED'
2025-11-21T10:35:57.1737389Z     'X-Ms-Client-Version': 'REDACTED'
2025-11-21T10:35:57.1737407Z     'api-supported-versions': 'REDACTED'
2025-11-21T10:35:57.1737427Z     'x-ms-client-request-id': 'dd03ce04-c6c5-11f0-b822-46a038d79753'
2025-11-21T10:35:57.1737446Z     'X-Microsoft-Skype-Chain-ID': 'REDACTED'
2025-11-21T10:35:57.1737465Z     'x-azure-ref': 'REDACTED'
2025-11-21T10:35:57.1737484Z     'Strict-Transport-Security': 'REDACTED'
2025-11-21T10:35:57.1737503Z     'X-Cache': 'REDACTED'
2025-11-21T10:35:57.1860892Z INFO:voicerag:✅ Call answered: 0e006880-2bae-4458-94ab-1261eaac1af3
2025-11-21T10:35:57.9391505Z INFO:voicerag:✅ Phone call activity created: dae90ed7-c5c6-f011-bbd3-7c1e52023626
2025-11-21T10:36:09.1671262Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallConnected
2025-11-21T10:36:09.5844388Z INFO:voicerag:📨 Callback event: Microsoft.Communication.ParticipantsUpdated
2025-11-21T10:36:30.6705673Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallDisconnected
2025-11-21T10:37:14.2064239Z INFO:voicerag:📩 Received data type: <class 'list'>
2025-11-21T10:37:14.207597Z INFO:voicerag:📋 Event type: Microsoft.Communication.IncomingCall
2025-11-21T10:37:14.2076153Z INFO:voicerag:📞 Processing incoming call
2025-11-21T10:37:14.2183461Z INFO:voicerag:Call data keys: dict_keys(['to', 'from', 'serverCallId', 'callerDisplayName', 'incomingCallContext', 'correlationId'])
2025-11-21T10:37:14.2183628Z INFO:voicerag:📞 Caller: +13463755076
2025-11-21T10:37:14.4461463Z INFO:voicerag:No customer found for +13463755076
2025-11-21T10:37:14.4584041Z INFO:voicerag:📞 Answering call...
2025-11-21T10:37:14.4584491Z INFO:azure.core.pipeline.policies.http_logging_policy:Request URL: 'https://bizapp-acs.unitedstates.communication.azure.com/calling/callConnections:answer?api-version=REDACTED'
2025-11-21T10:37:14.458452Z Request method: 'POST'
2025-11-21T10:37:14.4584539Z Request headers:
2025-11-21T10:37:14.4584558Z     'Content-Type': 'application/json'
2025-11-21T10:37:14.4584576Z     'Content-Length': '9053'
2025-11-21T10:37:14.4584594Z     'Repeatability-First-Sent': 'REDACTED'
2025-11-21T10:37:14.4584614Z     'Repeatability-Request-ID': 'REDACTED'
2025-11-21T10:37:14.4584632Z     'Accept': 'application/json'
2025-11-21T10:37:14.4584688Z     'x-ms-client-request-id': '0b57398a-c6c6-11f0-b822-46a038d79753'
2025-11-21T10:37:14.4584713Z     'User-Agent': 'azsdk-python-communication-callautomation/1.5.0 Python/3.12.12 (Linux-6.6.104.2-1.azl3-x86_64-with-glibc2.31)'
2025-11-21T10:37:14.4584731Z     'x-ms-date': 'REDACTED'
2025-11-21T10:37:14.458475Z     'x-ms-content-sha256': 'REDACTED'
2025-11-21T10:37:14.4584768Z     'x-ms-return-client-request-id': 'true'
2025-11-21T10:37:14.4584786Z     'Authorization': 'REDACTED'
2025-11-21T10:37:14.4584804Z A body is sent with the request
2025-11-21T10:37:14.7189932Z INFO:azure.core.pipeline.policies.http_logging_policy:Response status: 200
2025-11-21T10:37:14.7190369Z Response headers:
2025-11-21T10:37:14.7190404Z     'Date': 'Fri, 21 Nov 2025 10:37:14 GMT'
2025-11-21T10:37:14.7190428Z     'Content-Type': 'application/json; charset=utf-8'
2025-11-21T10:37:14.7190491Z     'Transfer-Encoding': 'chunked'
2025-11-21T10:37:14.7190508Z     'Connection': 'keep-alive'
2025-11-21T10:37:14.7190523Z     'MS-CV': 'REDACTED'
2025-11-21T10:37:14.7190539Z     'X-Microsoft-Skype-Client': 'REDACTED'
2025-11-21T10:37:14.7190555Z     'X-Ms-Client-Version': 'REDACTED'
2025-11-21T10:37:14.7190571Z     'api-supported-versions': 'REDACTED'
2025-11-21T10:37:14.7190588Z     'x-ms-client-request-id': '0b57398a-c6c6-11f0-b822-46a038d79753'
2025-11-21T10:37:14.7190605Z     'X-Microsoft-Skype-Chain-ID': 'REDACTED'
2025-11-21T10:37:14.7190622Z     'x-azure-ref': 'REDACTED'
2025-11-21T10:37:14.7190638Z     'Strict-Transport-Security': 'REDACTED'
2025-11-21T10:37:14.7190653Z     'X-Cache': 'REDACTED'
2025-11-21T10:37:14.7190698Z INFO:voicerag:✅ Call answered: 1f006980-d45f-4bf1-8f63-a394268e43f3
2025-11-21T10:37:15.1728006Z INFO:voicerag:✅ Phone call activity created: 701cb20a-c6c6-f011-bbd3-7c1e527fc4af
2025-11-21T10:37:17.1141817Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallConnected
2025-11-21T10:37:17.4704973Z INFO:voicerag:📨 Callback event: Microsoft.Communication.ParticipantsUpdated
2025-11-21T10:37:38.3519429Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallDisconnected
2025-11-21T10:38:58.9592922Z INFO:voicerag:📩 Received data type: <class 'list'>
2025-11-21T10:38:58.9603933Z INFO:voicerag:📋 Event type: Microsoft.Communication.IncomingCall
2025-11-21T10:38:58.960434Z INFO:voicerag:📞 Processing incoming call
2025-11-21T10:38:58.960438Z INFO:voicerag:Call data keys: dict_keys(['to', 'from', 'serverCallId', 'callerDisplayName', 'incomingCallContext', 'correlationId'])
2025-11-21T10:38:58.9604399Z INFO:voicerag:📞 Caller: +13463755076
2025-11-21T10:38:59.0158995Z INFO:voicerag:No customer found for +13463755076
2025-11-21T10:38:59.0195976Z INFO:voicerag:📞 Answering call...
2025-11-21T10:38:59.0196341Z INFO:azure.core.pipeline.policies.http_logging_policy:Request URL: 'https://bizapp-acs.unitedstates.communication.azure.com/calling/callConnections:answer?api-version=REDACTED'
2025-11-21T10:38:59.0196381Z Request method: 'POST'
2025-11-21T10:38:59.0196399Z Request headers:
2025-11-21T10:38:59.0196503Z     'Content-Type': 'application/json'
2025-11-21T10:38:59.0196523Z     'Content-Length': '9075'
2025-11-21T10:38:59.0196542Z     'Repeatability-First-Sent': 'REDACTED'
2025-11-21T10:38:59.0196561Z     'Repeatability-Request-ID': 'REDACTED'
2025-11-21T10:38:59.0196579Z     'Accept': 'application/json'
2025-11-21T10:38:59.01966Z     'x-ms-client-request-id': '49a9c6f8-c6c6-11f0-b822-46a038d79753'
2025-11-21T10:38:59.0196625Z     'User-Agent': 'azsdk-python-communication-callautomation/1.5.0 Python/3.12.12 (Linux-6.6.104.2-1.azl3-x86_64-with-glibc2.31)'
2025-11-21T10:38:59.0196644Z     'x-ms-date': 'REDACTED'
2025-11-21T10:38:59.0196663Z     'x-ms-content-sha256': 'REDACTED'
2025-11-21T10:38:59.0196681Z     'x-ms-return-client-request-id': 'true'
2025-11-21T10:38:59.01967Z     'Authorization': 'REDACTED'
2025-11-21T10:38:59.0196738Z A body is sent with the request
2025-11-21T10:38:59.2164957Z INFO:azure.core.pipeline.policies.http_logging_policy:Response status: 200
2025-11-21T10:38:59.2165694Z Response headers:
2025-11-21T10:38:59.2165734Z     'Date': 'Fri, 21 Nov 2025 10:38:59 GMT'
2025-11-21T10:38:59.2165755Z     'Content-Type': 'application/json; charset=utf-8'
2025-11-21T10:38:59.2165773Z     'Transfer-Encoding': 'chunked'
2025-11-21T10:38:59.2165792Z     'Connection': 'keep-alive'
2025-11-21T10:38:59.2165809Z     'MS-CV': 'REDACTED'
2025-11-21T10:38:59.2165834Z     'X-Microsoft-Skype-Client': 'REDACTED'
2025-11-21T10:38:59.2165855Z     'X-Ms-Client-Version': 'REDACTED'
2025-11-21T10:38:59.2165873Z     'api-supported-versions': 'REDACTED'
2025-11-21T10:38:59.2165989Z     'x-ms-client-request-id': '49a9c6f8-c6c6-11f0-b822-46a038d79753'
2025-11-21T10:38:59.2166009Z     'X-Microsoft-Skype-Chain-ID': 'REDACTED'
2025-11-21T10:38:59.2166028Z     'x-azure-ref': 'REDACTED'
2025-11-21T10:38:59.2166047Z     'Strict-Transport-Security': 'REDACTED'
2025-11-21T10:38:59.2166065Z     'X-Cache': 'REDACTED'
2025-11-21T10:38:59.2180915Z INFO:voicerag:✅ Call answered: 16005b80-f5dd-48ca-ae8d-264bb7262453
2025-11-21T10:38:59.3213879Z INFO:voicerag:✅ Phone call activity created: 7f9f3247-c6c6-f011-bbd3-7c1e527fc4af
2025-11-21T10:39:01.2078592Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallConnected
2025-11-21T10:39:01.482609Z INFO:voicerag:📨 Callback event: Microsoft.Communication.ParticipantsUpdated
2025-11-21T10:39:16.6981564Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallDisconnected
2025-11-21T10:39:46.2284501Z INFO:voicerag:📩 Received data type: <class 'list'>
2025-11-21T10:39:46.2331634Z INFO:voicerag:📋 Event type: Microsoft.Communication.IncomingCall
2025-11-21T10:39:46.233176Z INFO:voicerag:📞 Processing incoming call
2025-11-21T10:39:46.2331868Z INFO:voicerag:Call data keys: dict_keys(['to', 'from', 'serverCallId', 'callerDisplayName', 'incomingCallContext', 'correlationId'])
2025-11-21T10:39:46.2331898Z INFO:voicerag:📞 Caller: +13463755076
2025-11-21T10:39:46.4318736Z INFO:voicerag:No customer found for +13463755076
2025-11-21T10:39:46.4319421Z INFO:voicerag:📞 Answering call...
2025-11-21T10:39:46.437845Z INFO:azure.core.pipeline.policies.http_logging_policy:Request URL: 'https://bizapp-acs.unitedstates.communication.azure.com/calling/callConnections:answer?api-version=REDACTED'
2025-11-21T10:39:46.4378833Z Request method: 'POST'
2025-11-21T10:39:46.4378866Z Request headers:
2025-11-21T10:39:46.4378888Z     'Content-Type': 'application/json'
2025-11-21T10:39:46.4378908Z     'Content-Length': '9067'
2025-11-21T10:39:46.437893Z     'Repeatability-First-Sent': 'REDACTED'
2025-11-21T10:39:46.4378951Z     'Repeatability-Request-ID': 'REDACTED'
2025-11-21T10:39:46.4378973Z     'Accept': 'application/json'
2025-11-21T10:39:46.4378996Z     'x-ms-client-request-id': '65ece304-c6c6-11f0-b822-46a038d79753'
2025-11-21T10:39:46.4379022Z     'User-Agent': 'azsdk-python-communication-callautomation/1.5.0 Python/3.12.12 (Linux-6.6.104.2-1.azl3-x86_64-with-glibc2.31)'
2025-11-21T10:39:46.4379043Z     'x-ms-date': 'REDACTED'
2025-11-21T10:39:46.4379091Z     'x-ms-content-sha256': 'REDACTED'
2025-11-21T10:39:46.4379114Z     'x-ms-return-client-request-id': 'true'
2025-11-21T10:39:46.4379134Z     'Authorization': 'REDACTED'
2025-11-21T10:39:46.4379154Z A body is sent with the request
2025-11-21T10:39:46.4777145Z INFO:azure.core.pipeline.policies.http_logging_policy:Response status: 200
2025-11-21T10:39:46.4777574Z Response headers:
2025-11-21T10:39:46.4777608Z     'Date': 'Fri, 21 Nov 2025 10:39:46 GMT'
2025-11-21T10:39:46.4777636Z     'Content-Type': 'application/json; charset=utf-8'
2025-11-21T10:39:46.4777661Z     'Transfer-Encoding': 'chunked'
2025-11-21T10:39:46.4777683Z     'Connection': 'keep-alive'
2025-11-21T10:39:46.4777704Z     'MS-CV': 'REDACTED'
2025-11-21T10:39:46.4777807Z     'X-Microsoft-Skype-Client': 'REDACTED'
2025-11-21T10:39:46.4777832Z     'X-Ms-Client-Version': 'REDACTED'
2025-11-21T10:39:46.4777853Z     'api-supported-versions': 'REDACTED'
2025-11-21T10:39:46.4777877Z     'x-ms-client-request-id': '65ece304-c6c6-11f0-b822-46a038d79753'
2025-11-21T10:39:46.47779Z     'X-Microsoft-Skype-Chain-ID': 'REDACTED'
2025-11-21T10:39:46.4777921Z     'x-azure-ref': 'REDACTED'
2025-11-21T10:39:46.4777943Z     'Strict-Transport-Security': 'REDACTED'
2025-11-21T10:39:46.4777963Z     'X-Cache': 'REDACTED'
2025-11-21T10:39:46.4797004Z INFO:voicerag:✅ Call answered: 16005b80-1cd8-4664-bc7d-1b7c24f80729
2025-11-21T10:39:46.857555Z INFO:voicerag:✅ Phone call activity created: 35737865-c6c6-f011-bbd3-7c1e52813591
2025-11-21T10:39:51.5683252Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallConnected
2025-11-21T10:39:51.818821Z INFO:voicerag:📨 Callback event: Microsoft.Communication.ParticipantsUpdated
2025-11-21T10:40:02.3970626Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallDisconnected
2025-11-21T10:43:18.7642826Z INFO:voicerag:📩 Received data type: <class 'list'>
2025-11-21T10:43:18.7643398Z INFO:voicerag:📋 Event type: Microsoft.Communication.IncomingCall
2025-11-21T10:43:18.7643435Z INFO:voicerag:📞 Processing incoming call
2025-11-21T10:43:18.7643458Z INFO:voicerag:Call data keys: dict_keys(['to', 'from', 'serverCallId', 'callerDisplayName', 'incomingCallContext', 'correlationId'])
2025-11-21T10:43:18.7643477Z INFO:voicerag:📞 Caller: +13463755076
2025-11-21T10:43:18.8184518Z INFO:voicerag:No customer found for +13463755076
2025-11-21T10:43:18.8188352Z INFO:voicerag:📞 Answering call...
2025-11-21T10:43:18.8201191Z INFO:azure.core.pipeline.policies.http_logging_policy:Request URL: 'https://bizapp-acs.unitedstates.communication.azure.com/calling/callConnections:answer?api-version=REDACTED'
2025-11-21T10:43:18.8201445Z Request method: 'POST'
2025-11-21T10:43:18.8201468Z Request headers:
2025-11-21T10:43:18.8201539Z     'Content-Type': 'application/json'
2025-11-21T10:43:18.8201555Z     'Content-Length': '9061'
2025-11-21T10:43:18.8201572Z     'Repeatability-First-Sent': 'REDACTED'
2025-11-21T10:43:18.8201588Z     'Repeatability-Request-ID': 'REDACTED'
2025-11-21T10:43:18.8201606Z     'Accept': 'application/json'
2025-11-21T10:43:18.8201624Z     'x-ms-client-request-id': 'e4847538-c6c6-11f0-b822-46a038d79753'
2025-11-21T10:43:18.8201644Z     'User-Agent': 'azsdk-python-communication-callautomation/1.5.0 Python/3.12.12 (Linux-6.6.104.2-1.azl3-x86_64-with-glibc2.31)'
2025-11-21T10:43:18.8201772Z     'x-ms-date': 'REDACTED'
2025-11-21T10:43:18.8201792Z     'x-ms-content-sha256': 'REDACTED'
2025-11-21T10:43:18.8201809Z     'x-ms-return-client-request-id': 'true'
2025-11-21T10:43:18.8201825Z     'Authorization': 'REDACTED'
2025-11-21T10:43:18.8201864Z A body is sent with the request
2025-11-21T10:43:18.9850899Z INFO:azure.core.pipeline.policies.http_logging_policy:Response status: 200
2025-11-21T10:43:18.9851411Z Response headers:
2025-11-21T10:43:18.985144Z     'Date': 'Fri, 21 Nov 2025 10:43:18 GMT'
2025-11-21T10:43:18.9851464Z     'Content-Type': 'application/json; charset=utf-8'
2025-11-21T10:43:18.9851489Z     'Transfer-Encoding': 'chunked'
2025-11-21T10:43:18.9851508Z     'Connection': 'keep-alive'
2025-11-21T10:43:18.9851526Z     'MS-CV': 'REDACTED'
2025-11-21T10:43:18.9851546Z     'X-Microsoft-Skype-Client': 'REDACTED'
2025-11-21T10:43:18.9851565Z     'X-Ms-Client-Version': 'REDACTED'
2025-11-21T10:43:18.9851594Z     'api-supported-versions': 'REDACTED'
2025-11-21T10:43:18.9851699Z     'x-ms-client-request-id': 'e4847538-c6c6-11f0-b822-46a038d79753'
2025-11-21T10:43:18.9861561Z     'X-Microsoft-Skype-Chain-ID': 'REDACTED'
2025-11-21T10:43:18.986161Z     'x-azure-ref': 'REDACTED'
2025-11-21T10:43:18.9861629Z     'Strict-Transport-Security': 'REDACTED'
2025-11-21T10:43:18.9861645Z     'X-Cache': 'REDACTED'
2025-11-21T10:43:18.9862407Z INFO:voicerag:✅ Call answered: 29005c80-3b13-4d0c-8a0e-ddd7a319f905
2025-11-21T10:43:19.0632653Z INFO:voicerag:✅ Phone call activity created: 708beee2-c6c6-f011-bbd3-7c1e52813591
2025-11-21T10:43:20.3008628Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallConnected
2025-11-21T10:43:20.3097963Z INFO:voicerag:📨 Callback event: Microsoft.Communication.ParticipantsUpdated
2025-11-21T10:44:26.8986525Z INFO:voicerag:📨 Callback event: Microsoft.Communication.CallDisconnected
2025-11-21T11:47:06.124Z No new trace in the past 1 min(s).
2025-11-21T11:48:06.124Z No new trace in the past 2 min(s).
2025-11-21T11:49:06.124Z No new trace in the past 3 min(s).
