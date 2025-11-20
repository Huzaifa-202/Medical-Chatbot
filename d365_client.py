import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

logger = logging.getLogger("voicerag")

class D365Client:
    def __init__(self, tenant_id: str, client_id: str, client_secret: str, d365_url: str):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.d365_url = d365_url.rstrip('/')
        self.token = None
        self.token_expires = None
    
    async def get_access_token(self) -> str:
        """Get OAuth token for D365 API"""
        if self.token and self.token_expires and datetime.now() < self.token_expires:
            return self.token
        
        logger.info("Getting D365 access token...")
        
        token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': f'{self.d365_url}/.default'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    self.token = result['access_token']
                    expires_in = result.get('expires_in', 3600)
                    self.token_expires = datetime.now() + timedelta(seconds=expires_in - 300)
                    logger.info("✅ D365 token obtained")
                    return self.token
                else:
                    error = await response.text()
                    logger.error(f"❌ D365 token error: {error}")
                    raise Exception(f"Failed to get D365 token: {error}")
    
    async def lookup_contact_by_phone(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """Look up contact in D365 by phone number"""
        try:
            token = await self.get_access_token()
            
            # Clean phone number
            clean_phone = phone_number.replace('+', '').replace('-', '').replace(' ', '').replace('(', '').replace(')', '')
            
            headers = {
                'Authorization': f'Bearer {token}',
                'Accept': 'application/json',
                'OData-MaxVersion': '4.0',
                'OData-Version': '4.0'
            }
            
            # Search in multiple phone fields
            filter_query = f"contains(telephone1,'{clean_phone}') or contains(mobilephone,'{clean_phone}') or contains(telephone2,'{clean_phone}')"
            
            url = f"{self.d365_url}/api/data/v9.2/contacts?$filter={filter_query}&$select=contactid,fullname,emailaddress1,telephone1,mobilephone&$top=1"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('value') and len(data['value']) > 0:
                            contact = data['value'][0]
                            logger.info(f"✅ Found customer: {contact.get('fullname')}")
                            return contact
                        else:
                            logger.info(f"No customer found for {phone_number}")
                            return None
                    else:
                        error = await response.text()
                        logger.error(f"❌ Contact lookup error: {error}")
                        return None
        except Exception as e:
            logger.error(f"❌ Error in contact lookup: {e}")
            return None
    
    async def create_phone_call_activity(
        self, 
        caller_id: str, 
        contact_id: Optional[str],
        subject: str,
        description: str
    ) -> Optional[str]:
        """Create phone call activity in D365"""
        try:
            token = await self.get_access_token()
            
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'OData-MaxVersion': '4.0',
                'OData-Version': '4.0',
                'Prefer': 'return=representation'
            }
            
            # Create phone call activity
            activity_data = {
                "subject": subject,
                "description": description,
                "phonenumber": caller_id,
                "directioncode": False,  # False = Incoming
                "statecode": 0,  # 0 = Open
                "statuscode": 1  # 1 = Open
            }
            
            # Link to contact if found
            if contact_id:
                activity_data["regardingobjectid_contact@odata.bind"] = f"/contacts({contact_id})"
            
            url = f"{self.d365_url}/api/data/v9.2/phonecalls"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=activity_data) as response:
                    if response.status in [201, 204]:
                        # Get activity ID from response
                        if response.status == 201:
                            result = await response.json()
                            activity_id = result.get('activityid')
                        else:
                            # Parse from Location header
                            location = response.headers.get('OData-EntityId', '')
                            activity_id = location.split('(')[-1].split(')')[0] if '(' in location else None
                        
                        logger.info(f"✅ Phone call activity created: {activity_id}")
                        return activity_id
                    else:
                        error = await response.text()
                        logger.error(f"❌ Failed to create activity: {response.status} - {error}")
                        return None
        except Exception as e:
            logger.error(f"❌ Error creating activity: {e}")
            return None
    
    async def update_phone_call_duration(self, activity_id: str, duration_minutes: int):
        """Update phone call activity with duration"""
        try:
            token = await self.get_access_token()
            
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
                'OData-MaxVersion': '4.0',
                'OData-Version': '4.0'
            }
            
            update_data = {
                "actualdurationminutes": duration_minutes,
                "statecode": 1,  # 1 = Completed
                "statuscode": 2  # 2 = Made
            }
            
            url = f"{self.d365_url}/api/data/v9.2/phonecalls({activity_id})"
            
            async with aiohttp.ClientSession() as session:
                async with session.patch(url, headers=headers, json=update_data) as response:
                    if response.status in [204, 200]:
                        logger.info(f"✅ Activity updated with duration: {duration_minutes} min")
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"❌ Failed to update activity: {error}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error updating activity: {e}")
            return False
    
    async def add_note_to_activity(self, activity_id: str, note_text: str) -> bool:
        """Add note/transcript to phone call activity"""
        try:
            token = await self.get_access_token()
            
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            note_data = {
                "notetext": note_text,
                "subject": "Bot Conversation Transcript",
                "objectid_phonecall@odata.bind": f"/phonecalls({activity_id})"
            }
            
            url = f"{self.d365_url}/api/data/v9.2/annotations"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=note_data) as response:
                    if response.status in [201, 204]:
                        logger.info("✅ Note added to activity")
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"❌ Failed to add note: {error}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error adding note: {e}")
            return False
