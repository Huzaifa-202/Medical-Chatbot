import os
import time
import re
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, VectorizableTextQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# --- Load environment variables ---
load_dotenv()

# --- Azure Search Configuration ---
service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
api_key = os.getenv("AZURE_SEARCH_API_KEY")

# --- Azure OpenAI Configuration ---
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # e.g., "gpt-4o"
vector_field_name = os.getenv("AZURE_SEARCH_VECTOR_FIELD", "text_vector")  # Vector field in your index

# --- Initialize Clients ---
search_client = SearchClient(
    endpoint=service_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(api_key)
)

# Set required environment variables for SDK
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = openai_endpoint
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

client = AzureOpenAI()  # no arguments

# --- Helper: sanitize text ---
def sanitize_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --- Step 1: Retrieve from Azure Search using VECTOR SEARCH (OPTIMIZED) ---
def get_search_results(query_text: str, top_k: int = 5):
    """
    OPTIMIZED VECTOR SEARCH with better accuracy
    - Fetch more results (top 5) for better coverage
    - Azure AI Search handles embedding generation automatically
    """
    start_time = time.perf_counter()

    # Use VectorizableTextQuery - Azure AI Search will vectorize the query automatically
    vector_query = VectorizableTextQuery(
        text=query_text,
        k_nearest_neighbors=top_k,
        fields=vector_field_name  # Your vector field (e.g., "text_vector")
    )

    # Pure vector search (search_text=None means no keyword search)
    results = search_client.search(
        search_text=None,  # None = pure vector search, not keyword/semantic
        vector_queries=[vector_query],
        select=["title", "chunk"],  # Only fetch needed fields for speed
        top=top_k
    )

    # VERIFICATION: Print search type
    print(f"🔎 Search Type: VECTOR SEARCH (using field '{vector_field_name}')")

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    # Minimal processing - no sanitization during collection
    docs = []
    for r in results:
        docs.append({
            "title": r.get("title", "N/A"),
            "chunk": r.get("chunk", ""),
            "score": r.get("@search.score", 0)
        })
    return docs, latency_ms

# --- Step 2: Build an enhanced prompt for better accuracy ---
def build_enhanced_prompt(query: str, docs):
    """
    Enhanced prompt engineering for better accuracy
    - Uses top 3-4 most relevant docs
    - Includes title context for better understanding
    - Clear instructions to reduce hallucination
    """
    # Use top 3-4 docs based on scores
    top_docs = docs[:4] if len(docs) >= 4 else docs

    context_lines = []
    for i, d in enumerate(top_docs, 1):
        # Include title for better context
        title = d['title']
        # Use full chunks (not truncated) for better accuracy
        chunk = d['chunk'][:600] if len(d['chunk']) > 600 else d['chunk']  # Increased from 400
        context_lines.append(f"[Document {i}: {title}]\n{chunk}")

    context = "\n\n".join(context_lines)

    # Improved prompt for accuracy
    prompt = (
        f"Using ONLY the information from these documents, answer the question.\n"
        f"If the answer isn't in the documents, say 'I don't have enough information to answer this.'\n\n"
        f"Documents:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    return prompt, top_docs

# --- Step 3: Generate GPT response with STREAMING ---
def get_gpt_response_stream(prompt: str):
    first_token_latency = 0
    final_answer = ""
    total_start_time = time.perf_counter()

    print("\n🤖 GPT Response (streaming):\n")
    try:
        stream = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You're a concise and factual assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3,
            stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                if first_token_latency == 0:
                    first_token_latency = time.perf_counter() - total_start_time
                print(delta.content, end="", flush=True)
                final_answer += delta.content

        total_end_time = time.perf_counter()
        total_latency_ms = (total_end_time - total_start_time) * 1000

        return final_answer, total_latency_ms, first_token_latency * 1000

    except Exception as e:
        err_str = str(e)
        if "content_filter" in err_str:
            print("\n⚠️ GPT Response was blocked by Azure content filter.")
        else:
            print(f"\n⚠️ GPT Error: {err_str}")
        return None, 0, 0

# --- Step 4: Run OPTIMIZED RAG pipeline ---
def run_rag_pipeline(query: str):
    overall_start = time.perf_counter()

    print(f"\n❓ Query: {query}")
    print("-" * 60)

    # Step 1: Vector search
    docs, search_latency = get_search_results(query)
    print(f"🔍 Retrieved {len(docs)} docs via VECTOR SEARCH in {search_latency:.2f} ms")

    # Step 2: Build enhanced prompt for better accuracy
    prompt, docs_used = build_enhanced_prompt(query, docs)

    # Step 3: Get GPT response with streaming
    answer, gpt_latency, first_token_latency = get_gpt_response_stream(prompt)

    overall_end = time.perf_counter()
    total_latency = (overall_end - overall_start) * 1000

    print("\n\n📊 Latency Breakdown:")
    print(f"├── Vector Search: {search_latency:.2f} ms")
    print(f"├── GPT First Token: {first_token_latency:.2f} ms")
    print(f"├── GPT Total: {gpt_latency:.2f} ms")
    print(f"└── Total End-to-End: {total_latency:.2f} ms")

    # Show retrieved context (compact)
    if docs_used:
        print(f"\n📄 Retrieved {len(docs_used)} documents (scores: {[f'{d["score"]:.3f}' for d in docs_used[:3]]})")

    return answer

# --- Main Chat Loop ---
if __name__ == "__main__":
    print("Welcome to the RAG assistant! Type 'exit' to quit.")
    while True:
        user_query = input("\nEnter your query: ")
        if user_query.lower() in ("exit", "quit"):
            print("Exiting. Goodbye!")
            break
        run_rag_pipeline(user_query)
