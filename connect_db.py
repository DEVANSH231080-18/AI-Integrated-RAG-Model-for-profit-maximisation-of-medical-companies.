import os 
from pymongo import MongoClient
from collections import defaultdict
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

client = MongoClient("mongodb://localhost:27017/")
db = client["raw_data"]

cpt_data = list(db["telMed2201.cpts"].find({}, {"_id": 1, "cptCode": 1, "price": 1}))
notes = list(db["doctor_cpt_notes"].find({}, {"_id": 0}))
providers = list(db["telMed2201.doctors"].find({}, {"_id": 0}))


cpt_id_map = {
    str(c["_id"]): {
        "cptCode": c.get("cptCode"),
        "price": c.get("price")
    }
    for c in cpt_data
}
provider_cpt_map = defaultdict(list)
provider_notes_map = defaultdict(list)

for n in notes:
    provider_id = str(n["doctor_id"])
    cpt = str(n["cptCode"])
    
    cpt_info = cpt_id_map.get(cpt, {})
    actual_cpt_code = cpt_info.get("cptCode", "NOT_FOUND")
    price = cpt_info.get("price", "NOT_FOUND")

    provider_cpt_map[provider_id].append({
        "cptCode": actual_cpt_code,
        "price": price
    })
    provider_notes_map[provider_id].append(n["doctorNotes"])

documents = []

for provider_id in provider_cpt_map:
    cpts = provider_cpt_map[provider_id]
    notes_text = " ".join(str(note) for note in provider_notes_map[provider_id])

    total_revenue = sum(c.get("price", 0) for c in cpts)
    cpt_codes = [c.get("cptCode", "NOT_FOUND") for c in cpts]

    doc = f"""
Provider: {provider_id}

DoctorNotes:
{notes_text}

CPTs frequently used:
{', '.join(cpt_codes)}

Total revenue potential:
{total_revenue}
"""
    documents.append(doc)

persist_directory = "./chroma_db"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)
doc_ids = [f"provider_{i}" for i in range(len(documents))]

if os.path.exists(persist_directory):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    existing = vectorstore.get()
    existing_ids = set(existing["ids"]) if existing["ids"] else set()

    new_texts = []
    new_ids = []

    for doc_id, doc in zip(doc_ids, documents):
        if doc_id not in existing_ids:
            new_ids.append(doc_id)
            new_texts.append(doc)

    if new_texts:
        vectorstore.add_texts(texts=new_texts, ids=new_ids)
        print("New embeddings added:", len(new_texts))
    else:
        print("All embeddings already exist. Nothing added.")

    # print("Total vectors in DB:", vectorstore._collection.count())

else:
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        ids=doc_ids,
        persist_directory=persist_directory
    )
    # print("Embeddings stored first time.")
    # print("Total vectors in DB:", vectorstore._collection.count())
# if all_data["ids"]:
#     vectorstore.delete(ids=all_data["ids"])
#     print("All vectors deleted.")
# else:
#     print("No vectors found.")
doc_id = None
user_query = None

def set_values_and_process(d_id, d_notes):
    global doc_id, user_query

    doc_id = d_id
    user_query = d_notes
    return {
        "doc_id": doc_id,
        "user_query": user_query,
    }
   
query = f"Find cptCode, price for doctor_notes that are similar to {user_query}"
results = vectorstore.similarity_search_with_score(query)
