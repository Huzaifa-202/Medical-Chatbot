import logging
from datetime import datetime
from aiohttp import web
from azure.communication.callautomation import CallAutomationClient
from azure.communication.callautomation.models import (
    MediaStreamingConfiguration,
    MediaStreamingAudioChannelType,
    MediaStreamingContentType,
    MediaStreamingTransportType
)

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
        """Handle incoming call from ACS"""
        try:
            data = await request.json()
            logger.info("📞 Incoming call received")
            
            incoming_call_context = data.get("incomingCallContext")
            caller_id = data.get("from", {}).get("phoneNumber", {}).get("value", "Unknown")
            
            logger.info(f"Caller: {caller_id}")
            
            # Look up customer in D365
            contact = None
            if self.d365_client:
                contact = await self.d365_client.lookup_contact_by_phone(caller_id)
            
            # Answer the call
            answer_result = self.client.answer_call(
                incoming_call_context=incoming_call_context,
                callback_url=f"{self.app_url}/api/callbacks"
            )
            
            call_connection_id = answer_result.call_connection_id
            logger.info(f"✅ Call answered: {call_connection_id}")
            
            # Get call connection
            call_connection = self.client.get_call_connection(call_connection_id)
            
            # Start media streaming to bot
            streaming_config = MediaStreamingConfiguration(
                transport_url=f"wss://{self.app_url.replace('https://', '').replace('http://', '')}/realtime?callId={call_connection_id}",
                transport_type=MediaStreamingTransportType.WEBSOCKET,
                content_type=MediaStreamingContentType.AUDIO,
                audio_channel_type=MediaStreamingAudioChannelType.MIXED
            )
            
            call_connection.start_media_streaming(streaming_config)
            logger.info("🎵 Audio streaming started")
            
            # Create activity in D365
            activity_id = None
            if self.d365_client:
                subject = f"Inbound Call - {contact['fullname'] if contact else caller_id}"
                description = f"Voice bot call started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                activity_id = await self.d365_client.create_phone_call_activity(
                    caller_id=caller_id,
                    contact_id=contact['contactid'] if contact else None,
                    subject=subject,
                    description=description
                )
            
            # Store call info
            active_calls[call_connection_id] = {
                "caller_id": caller_id,
                "contact": contact,
                "call_connection": call_connection,
                "start_time": datetime.now(),
                "activity_id": activity_id
            }
            
            call_transcripts[call_connection_id] = []
            
            return web.Response(status=200, text="Call answered")
            
        except Exception as e:
            logger.error(f"❌ Error handling call: {e}")
            return web.Response(status=500, text=str(e))
    
    async def handle_callbacks(self, request: web.Request):
        """Handle call events from ACS"""
        try:
            data = await request.json()
            
            for event in data:
                event_type = event.get("type")
                call_id = event.get("callConnectionId")
                
                logger.info(f"📨 Event: {event_type}")
                
                if event_type == "Microsoft.Communication.CallDisconnected":
                    await self.handle_call_end(call_id)
                
                elif event_type == "Microsoft.Communication.MediaStreamingStarted":
                    logger.info("✅ Media streaming active")
                
                elif event_type == "Microsoft.Communication.MediaStreamingStopped":
                    logger.info("⏹️ Media streaming stopped")
            
            return web.Response(status=200)
            
        except Exception as e:
            logger.error(f"Error in callback: {e}")
            return web.Response(status=500)
    
    async def handle_call_end(self, call_connection_id: str):
        """Handle call end - save to D365"""
        if call_connection_id not in active_calls:
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
            # Update duration
            await self.d365_client.update_phone_call_duration(
                call_info['activity_id'],
                duration_minutes
            )
            
            # Add transcript
            if transcript:
                await self.d365_client.add_note_to_activity(
                    call_info['activity_id'],
                    transcript_text
                )
            
            logger.info("✅ Call logged to D365")
        
        # Cleanup
        del active_calls[call_connection_id]
        if call_connection_id in call_transcripts:
            del call_transcripts[call_connection_id]




(.venv) D:\One Drive\OneDrive - Systems Limited\Desktop\vb\aisearch-openai-rag-audio>python app/backend/app.py
Traceback (most recent call last):
  File "D:\One Drive\OneDrive - Systems Limited\Desktop\vb\aisearch-openai-rag-audio\app\backend\app.py", line 11, in <module>
    from acs_handler import ACSCallHandler  # NEW: Import ACS handler
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\One Drive\OneDrive - Systems Limited\Desktop\vb\aisearch-openai-rag-audio\app\backend\acs_handler.py", line 5, in <module>
    from azure.communication.callautomation import (
ImportError: cannot import name 'MediaStreamingConfiguration' from 'azure.communication.callautomation' (D:\One Drive\OneDrive - Systems Limited\Desktop\vb\aisearch-openai-rag-audio\.venv\Lib\site-packages\azure\communication\callautomation\__init__.py)

