import logging
from datetime import datetime
from aiohttp import web
from azure.communication.callautomation import CallAutomationClient

logger = logging.getLogger("voicerag")

# Store active calls
active_calls = {}
call_transcripts = {}

class ACSCallHandler:
    def __init__(self, connection_string: str, app_url: str, d365_client=None):
        self.client = CallAutomationClient.from_connection_string(connection_string)
        self.app_url = app_url
        self.d365_client = d365_client
    
    async def handle_incoming_call(self, request: web.Request):
        """Handle incoming call from ACS AND Event Grid validation"""
        try:
            data = await request.json()
            logger.info(f"📩 Received data type: {type(data)}")
            
            # Handle Event Grid format (array of events)
            if isinstance(data, list):
                if len(data) == 0:
                    return web.Response(status=200, text="Empty event array")
                
                event = data[0]
                event_type = event.get("eventType")
                
                logger.info(f"📋 Event type: {event_type}")
                
                # Event Grid subscription validation
                if event_type == "Microsoft.EventGrid.SubscriptionValidationEvent":
                    validation_code = event["data"]["validationCode"]
                    logger.info("✅ Event Grid validation - responding")
                    return web.json_response({"validationResponse": validation_code})
                
                # Handle incoming call event
                if event_type == "Microsoft.Communication.IncomingCall":
                    return await self._handle_call(event["data"])
                
                logger.warning(f"⚠️ Unknown event type: {event_type}")
                return web.Response(status=200, text="Event received")
            
            # Direct call (non-Event Grid format)
            else:
                return await self._handle_call(data)
                
        except Exception as e:
            logger.error(f"❌ Error handling request: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return web.Response(status=500, text=str(e))
    
    async def _handle_call(self, data):
        """Process incoming call - with audio streaming"""
        try:
            logger.info("📞 Processing incoming call")
            
            incoming_call_context = data.get("incomingCallContext")
            
            if not incoming_call_context:
                logger.error(f"❌ No incomingCallContext")
                return web.Response(status=400, text="Invalid call data")
            
            # Extract caller ID
            from_data = data.get("from", {})
            if isinstance(from_data, dict):
                phone_number_data = from_data.get("phoneNumber", {})
                if isinstance(phone_number_data, dict):
                    caller_id = phone_number_data.get("value", "Unknown")
                else:
                    caller_id = "Unknown"
            else:
                caller_id = "Unknown"
            
            logger.info(f"📞 Caller: {caller_id}")
            
            # Look up in D365 (optional)
            contact = None
            if self.d365_client and caller_id != "Unknown":
                contact = await self.d365_client.lookup_contact_by_phone(caller_id)
            
            # ✅ CONFIGURE MEDIA STREAMING OPTIONS
            media_streaming_options = None
            try:
                # Import inside try block to handle if not available
                from azure.communication.callautomation._models import (
                    MediaStreamingOptions,
                    MediaStreamingTransportType,
                    MediaStreamingContentType,
                    MediaStreamingAudioChannelType,
                    AudioFormat
                )
                
                # WebSocket URL for audio streaming
                websocket_url = f"wss://{self.app_url.replace('https://', '').replace('http://', '')}/realtime"
                
                logger.info(f"🎵 Configuring audio streaming to: {websocket_url}")
                
                # Create media streaming options using string values
                media_streaming_options = {
                    "transport_url": websocket_url,
                    "transport_type": "websocket",
                    "content_type": "audio",
                    "audio_channel_type": "mixed",
                    "start_media_streaming": True,
                    "enable_bidirectional": True,
                    "audio_format": "Pcm24KMono"
                }
                
                logger.info("✅ Media streaming options configured")
                
            except ImportError as e:
                logger.error(f"⚠️ Media streaming classes not available: {e}")
                logger.info("   Call will be answered without streaming")
            except Exception as e:
                logger.error(f"❌ Failed to configure streaming: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # ⚡ Answer the call WITH streaming options
            logger.info("📞 Answering call...")
            
            if media_streaming_options:
                answer_result = self.client.answer_call(
                    incoming_call_context=incoming_call_context,
                    callback_url=f"{self.app_url}/api/callbacks",
                    media_streaming=media_streaming_options
                )
            else:
                # Answer without streaming if config failed
                answer_result = self.client.answer_call(
                    incoming_call_context=incoming_call_context,
                    callback_url=f"{self.app_url}/api/callbacks"
                )
            
            call_connection_id = answer_result.call_connection_id
            
            if media_streaming_options:
                logger.info(f"✅ Call answered with streaming: {call_connection_id}")
            else:
                logger.info(f"✅ Call answered (no streaming): {call_connection_id}")
            
            # Create D365 activity (async, non-blocking)
            activity_id = None
            if self.d365_client:
                try:
                    subject = f"Inbound Call - {contact['fullname'] if contact else caller_id}"
                    description = f"Voice bot call started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    activity_id = await self.d365_client.create_phone_call_activity(
                        caller_id=caller_id,
                        contact_id=contact['contactid'] if contact else None,
                        subject=subject,
                        description=description
                    )
                    logger.info(f"✅ Phone call activity created: {activity_id}")
                except Exception as e:
                    logger.error(f"⚠️ Failed to create D365 activity: {e}")
            
            # Store call info
            active_calls[call_connection_id] = {
                "caller_id": caller_id,
                "contact": contact,
                "start_time": datetime.now(),
                "activity_id": activity_id
            }
            
            call_transcripts[call_connection_id] = []
            
            return web.Response(status=200, text="Call answered")
            
        except Exception as e:
            logger.error(f"❌ Error in _handle_call: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return web.Response(status=500, text=str(e))
    
    async def handle_callbacks(self, request: web.Request):
        """Handle call events from ACS"""
        try:
            data = await request.json()
            
            # Handle Event Grid validation for callbacks
            if isinstance(data, list) and len(data) > 0:
                event = data[0]
                if event.get("eventType") == "Microsoft.EventGrid.SubscriptionValidationEvent":
                    validation_code = event["data"]["validationCode"]
                    logger.info("✅ Event Grid validation for callbacks")
                    return web.json_response({"validationResponse": validation_code})
            
            # Handle regular callback events
            events = data if isinstance(data, list) else [data]
            
            for event in events:
                event_type = event.get("type") or event.get("eventType")
                call_id = event.get("callConnectionId")
                
                logger.info(f"📨 Callback event: {event_type}")
                
                if event_type and "CallDisconnected" in event_type:
                    if call_id:
                        await self.handle_call_end(call_id)
            
            return web.Response(status=200)
            
        except Exception as e:
            logger.error(f"❌ Error in callback: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return web.Response(status=500)
    
    async def handle_call_end(self, call_connection_id: str):
        """Handle call end - save to D365"""
        if call_connection_id not in active_calls:
            logger.warning(f"⚠️ Call {call_connection_id} not found in active calls")
            return
        
        call_info = active_calls[call_connection_id]
        transcript = call_transcripts.get(call_connection_id, [])
        
        logger.info(f"☎️ Call ended: {call_connection_id}")
        
        # Calculate duration
        duration_seconds = (datetime.now() - call_info['start_time']).total_seconds()
        duration_minutes = int(duration_seconds / 60)
        
        # Format transcript
        transcript_text = "=== Voice Bot Conversation ===\n\n"
        for turn in transcript:
            transcript_text += f"[{turn['timestamp']}]\n"
            transcript_text += f"{turn['speaker']}: {turn['text']}\n\n"
        
        transcript_text += f"\n=== Call Details ===\n"
        transcript_text += f"Duration: {duration_minutes} minutes\n"
        transcript_text += f"Caller: {call_info['caller_id']}\n"
        if call_info.get('contact'):
            transcript_text += f"Customer: {call_info['contact']['fullname']}\n"
        
        # Update D365
        if self.d365_client and call_info.get('activity_id'):
            try:
                await self.d365_client.update_phone_call_duration(
                    call_info['activity_id'],
                    duration_minutes
                )
                
                if transcript:
                    await self.d365_client.add_note_to_activity(
                        call_info['activity_id'],
                        transcript_text
                    )
                
                logger.info("✅ Call logged to D365")
            except Exception as e:
                logger.error(f"⚠️ Failed to update D365: {e}")
        
        # Cleanup
        del active_calls[call_connection_id]
        if call_connection_id in call_transcripts:
            del call_transcripts[call_connection_id]
