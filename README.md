
annotated-types==0.7.0
anyio==4.11.0
azure-common==1.1.28
azure-core==1.36.0
azure-search-documents==11.7.0b1
certifi==2025.10.5
charset-normalizer==3.4.4
colorama==0.4.6
distro==1.9.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.11
isodate==0.7.2
jiter==0.11.1
numpy==2.3.4
openai==2.6.0
pydantic==2.12.3
pydantic_core==2.41.4
python-dotenv==1.1.1
requests==2.32.5
sniffio==1.3.1
tqdm==4.67.1
typing-inspection==0.4.2
typing_extensions==4.15.0
urllib3==2.5.0



{
  "@odata.etag": "\"0x8DE14C351114E04\"",
  "name": "semanticsearchindex",
  "fields": [
    {
      "name": "chunk_id",
      "type": "Edm.String",
      "searchable": true,
      "filterable": false,
      "retrievable": true,
      "stored": true,
      "sortable": true,
      "facetable": false,
      "key": true,
      "analyzer": "keyword",
      "synonymMaps": []
    },
    {
      "name": "parent_id",
      "type": "Edm.String",
      "searchable": false,
      "filterable": true,
      "retrievable": true,
      "stored": true,
      "sortable": false,
      "facetable": false,
      "key": false,
      "synonymMaps": []
    },
    {
      "name": "chunk",
      "type": "Edm.String",
      "searchable": true,
      "filterable": false,
      "retrievable": true,
      "stored": true,
      "sortable": false,
      "facetable": false,
      "key": false,
      "synonymMaps": []
    },
    {
      "name": "title",
      "type": "Edm.String",
      "searchable": true,
      "filterable": false,
      "retrievable": true,
      "stored": true,
      "sortable": false,
      "facetable": false,
      "key": false,
      "synonymMaps": []
    },
    {
      "name": "text_vector",
      "type": "Collection(Edm.Single)",
      "searchable": true,
      "filterable": false,
      "retrievable": true,
      "stored": true,
      "sortable": false,
      "facetable": false,
      "key": false,
      "dimensions": 1536,
      "vectorSearchProfile": "semanticsearchindex-aiFoundryCatalog-text-profile",
      "synonymMaps": []
    }
  ],
  "scoringProfiles": [],
  "suggesters": [],
  "analyzers": [],
  "normalizers": [],
  "tokenizers": [],
  "tokenFilters": [],
  "charFilters": [],
  "similarity": {
    "@odata.type": "#Microsoft.Azure.Search.BM25Similarity"
  },
  "semantic": {
    "defaultConfiguration": "semanticsearchindex-semantic-configuration",
    "configurations": [
      {
        "name": "semanticsearchindex-semantic-configuration",
        "flightingOptIn": false,
        "rankingOrder": "BoostedRerankerScore",
        "prioritizedFields": {
          "titleField": {
            "fieldName": "title"
          },
          "prioritizedContentFields": [
            {
              "fieldName": "chunk"
            }
          ],
          "prioritizedKeywordsFields": []
        }
      }
    ]
  },
  "vectorSearch": {
    "algorithms": [
      {
        "name": "semanticsearchindex-algorithm",
        "kind": "hnsw",
        "hnswParameters": {
          "metric": "cosine",
          "m": 4,
          "efConstruction": 400,
          "efSearch": 500
        }
      }
    ],
    "profiles": [
      {
        "name": "semanticsearchindex-aiFoundryCatalog-text-profile",
        "algorithm": "semanticsearchindex-algorithm",
        "vectorizer": "semanticsearchindex-aiFoundryCatalog-text-vectorizer"
      }
    ],
    "vectorizers": [
      {
        "name": "semanticsearchindex-aiFoundryCatalog-text-vectorizer",
        "kind": "azureOpenAI",
        "azureOpenAIParameters": {
          "resourceUri": "https://ai-alikhuzema9041ai836005646697.openai.azure.com",
          "deploymentId": "text-embedding-3-small",
          "apiKey": "<redacted>",
          "modelName": "text-embedding-3-small"
        }
      }
    ],
    "compressions": []
  }
}
