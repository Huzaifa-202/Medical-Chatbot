async def _handle_call(self, data):
        """Process the actual incoming call"""
        try:
            logger.info("📞 Processing incoming call")
            
            # Extract the actual call data from Event Grid format
            if isinstance(data, dict):
                call_data = data.get("data") or data
            else:
                call_data = data
            
            incoming_call_context = call_data.get("incomingCallContext")
            
            if not incoming_call_context:
                logger.error("❌ No incomingCallContext found in data")
                return web.Response(status=400, text="Invalid call data")
            
            from_data = call_data.get("from", {})
            phone_number_data = from_data.get("phoneNumber", {}) if isinstance(from_data, dict) else {}
            caller_id = phone_number_data.get("value", "Unknown") if isinstance(phone_number_data, dict) else "Unknown"
            
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
