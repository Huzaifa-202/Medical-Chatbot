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
        
        # ⚡ Answer the call FIRST
        logger.info("📞 Answering call...")
        answer_result = self.client.answer_call(
            incoming_call_context=incoming_call_context,
            callback_url=f"{self.app_url}/api/callbacks"
        )
        
        call_connection_id = answer_result.call_connection_id
        logger.info(f"✅ Call answered: {call_connection_id}")
        
        # ⚡ NOW START AUDIO STREAMING
        try:
            from azure.communication.callautomation import (
                StartMediaStreamingOptions,
                MediaStreamingTransportType,
                MediaStreamingContentType,
                MediaStreamingAudioChannelType
            )
            
            call_connection = self.client.get_call_connection(call_connection_id)
            
            # Start media streaming to our /realtime endpoint
            websocket_url = f"wss://{self.app_url.replace('https://', '').replace('http://', '')}/realtime"
            
            logger.info(f"🎵 Starting audio streaming to: {websocket_url}")
            
            # Create proper streaming options
            streaming_options = StartMediaStreamingOptions(
                transport_url=websocket_url,
                transport_type=MediaStreamingTransportType.WEBSOCKET,
                content_type=MediaStreamingContentType.AUDIO,
                audio_channel_type=MediaStreamingAudioChannelType.MIXED
            )
            
            # Start streaming
            call_connection.start_media_streaming(streaming_options)
            
            logger.info("✅ Audio streaming started!")
            
        except Exception as e:
            logger.error(f"❌ Failed to start audio streaming: {e}")
            logger.error(f"   This means customer will hear silence!")
            import traceback
            logger.error(traceback.format_exc())
        
        # Create D365 activity (async, non-blocking)
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
