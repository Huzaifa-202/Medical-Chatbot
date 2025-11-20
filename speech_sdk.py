import logging
import os
from pathlib import Path
from aiohttp import web
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureDeveloperCliCredential, DefaultAzureCredential
from dotenv import load_dotenv
from ragtools import attach_rag_tools
from rtmt import RTMiddleTier
from d365_client import D365Client  # NEW: Import D365 client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voicerag")

async def create_app():
    if not os.environ.get("RUNNING_IN_PRODUCTION"):
        logger.info("Running in development mode, loading from .env file")
        load_dotenv()
    
    llm_key = os.environ.get("AZURE_OPENAI_API_KEY")
    search_key = os.environ.get("AZURE_SEARCH_API_KEY")
    credential = None
    
    if not llm_key or not search_key:
        if tenant_id := os.environ.get("AZURE_TENANT_ID"):
            logger.info("Using AzureDeveloperCliCredential with tenant_id %s", tenant_id)
            credential = AzureDeveloperCliCredential(tenant_id=tenant_id, process_timeout=60)
        else:
            logger.info("Using DefaultAzureCredential")
            credential = DefaultAzureCredential()
    
    llm_credential = AzureKeyCredential(llm_key) if llm_key else credential
    search_credential = AzureKeyCredential(search_key) if search_key else credential
    
    app = web.Application()
    
    # NEW: Setup D365 integration
    d365_client = None
    if os.environ.get("D365_TENANT_ID"):
        try:
            d365_client = D365Client(
                tenant_id=os.environ["D365_TENANT_ID"],
                client_id=os.environ["D365_CLIENT_ID"],
                client_secret=os.environ["D365_CLIENT_SECRET"],
                d365_url=os.environ["D365_URL"]
            )
            logger.info("✅ D365 integration enabled")
        except Exception as e:
            logger.error(f"❌ D365 setup failed: {e}")
    else:
        logger.warning("⚠️ D365 credentials not found in environment")
    
    rtmt = RTMiddleTier(
        credentials=llm_credential,
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        deployment=os.environ["AZURE_OPENAI_REALTIME_DEPLOYMENT"],
        voice_choice=os.environ.get("AZURE_OPENAI_REALTIME_VOICE_CHOICE") or "alloy"
    )
    
    rtmt.system_message = """
You are a helpful, concise voice assistant for FAQs. Always give answers based only on information from the knowledge base using the 'search' tool. 
Give answer in the same language as of user. If user switches language mid conversation You must switch your language according to user language.
Use these rules strictly:  
1. Always check the knowledge base with the 'search' tool before answering.  
3. Keep answers extremely short, ideally a single sentence, as the user listens via audio.  
4. Do not read file names, keys, or source paths aloud.  
5. Context matters: remember the user may ask follow-up questions about the same location or service; avoid unnecessary repetition.  
6. If the knowledge base has no answer, respond: "I don't know."  
Example interaction logic:  
- User asks nearest station → ask for location if not provided.  
- Just give the station name which is nearer to customer.Don't give information other than station name.I just want name of the station.
- User asks about services → provide concise yes/no.  
- User asks about items → provide short availability info.  
""".strip()
    
    attach_rag_tools(rtmt,
        credentials=search_credential,
        search_endpoint=os.environ.get("AZURE_SEARCH_ENDPOINT"),
        search_index=os.environ.get("AZURE_SEARCH_INDEX"),
        semantic_configuration=os.environ.get("AZURE_SEARCH_SEMANTIC_CONFIGURATION") or "default",
        identifier_field=os.environ.get("AZURE_SEARCH_IDENTIFIER_FIELD") or "chunk_id",
        content_field=os.environ.get("AZURE_SEARCH_CONTENT_FIELD") or "chunk",
        embedding_field=os.environ.get("AZURE_SEARCH_EMBEDDING_FIELD") or "text_vector",
        title_field=os.environ.get("AZURE_SEARCH_TITLE_FIELD") or "title",
        use_vector_query=(os.environ.get("AZURE_SEARCH_USE_VECTOR_QUERY") == "true") or True
    )
    
    rtmt.attach_to_app(app, "/realtime")
    
    current_directory = Path(__file__).parent
    app.add_routes([web.get('/', lambda _: web.FileResponse(current_directory / 'static/index.html'))])
    app.router.add_static('/', path=current_directory / 'static', name='static')
    
    return app

if __name__ == "__main__":
    host = "localhost"
    port = 8765
    web.run_app(create_app(), host=host, port=port)
