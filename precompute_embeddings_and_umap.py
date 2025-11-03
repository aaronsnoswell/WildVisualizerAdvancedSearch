import json
import keras
import random
import os
import sqlite3
from openai import OpenAI
import numpy as np
import joblib
from umap.parametric_umap import ParametricUMAP
import tiktoken
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from tqdm import tqdm
import gzip
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import threading

# Thread lock for database operations
db_lock = threading.Lock()

n_per_language = 50000  # Adjust as needed
LANGUAGES = ['all', 'english', 'chinese', 'russian', 'spanish', 'french', 'portuguese', 'german', 'italian', 'turkish', 'arabic', 'japanese', 'korean', 'polish', 'vietnamese']

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
tokenizer = tiktoken.get_encoding('cl100k_base')


def gzip_file(input_file, output_file):
    with open(input_file, 'rb') as f_in:
        with gzip.open(output_file, 'wb') as f_out:
            f_out.writelines(f_in)

def create_database(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cache
                 (key TEXT PRIMARY KEY, prompt TEXT, embedding TEXT)''')
    conn.commit()
    conn.close()
    return db_name

def insert_or_update(db_name, key, prompt, embedding):
    """Thread safe version of insert_or_update()"""
    with db_lock:  # Add lock
        conn = sqlite3.connect(db_name, timeout=30.0)  # Add timeout
        conn.execute(
            "PRAGMA journal_mode=WAL"
        )  # Enable WAL mode for better concurrency
        c = conn.cursor()
        c.execute(
            """INSERT OR REPLACE INTO cache
                     (key, prompt, embedding) VALUES (?, ?, ?)""",
            (key, prompt, json.dumps(embedding)),
        )
        conn.commit()
        conn.close()


# def retrieve(db_name, key):
#     conn = sqlite3.connect(db_name)
#     c = conn.cursor()
#     c.execute("SELECT prompt, embedding FROM cache WHERE key=?", (key,))
#     result = c.fetchone()
#     conn.close()
#     if result:
#         return True, json.loads(result[1])
#     else:
#         return False, None


def retrieve(db_name, key):
    """Thread safe version of retrieve()"""
    with db_lock:  # Add lock
        conn = sqlite3.connect(db_name, timeout=30.0)  # Add timeout
        c = conn.cursor()
        c.execute("SELECT prompt, embedding FROM cache WHERE key=?", (key,))
        result = c.fetchone()
        conn.close()
        if result:
            return True, json.loads(result[1])
        else:
            return False, None


def get_embedding_with_cache(database_name, conversation_id, prompt, model='text-embedding-3-small'):
    key = conversation_id
    hit, embedding = retrieve(database_name, key)
    if not hit:
        tokens = tokenizer.encode(prompt, disallowed_special=())
        #if len(tokens) > 8192:
        #    tokens = tokens[:8192]
        #    prompt = tokenizer.decode(tokens)
        if len(tokens) > 8100:
            tokens = tokens[:8100]
            prompt = tokenizer.decode(tokens)
        embedding = client.embeddings.create(input=[prompt], model=model).data[0].embedding
        insert_or_update(database_name, key, json.dumps(prompt), embedding)
    #else:
    #    print('Cache hit for embedding')
    return embedding

def conditional_reservoir_sample(dataset, n, language=None):
    reservoir = []
    count = 0
    seen = set([])
    for item in tqdm(dataset, desc=f"Sampling {language if language else 'all'}"):
        if language is None or item['language'].lower() == language.lower():
            first_turn = item['conversation'][0]['content'].strip()
            if not first_turn:
                continue
            if first_turn in seen:
                continue
            seen.add(first_turn)
            new_item = {}
            for key in item:
                if key in ['conversation', 'conversation_id']:
                    new_item[key] = item[key]
            item = new_item
            item['conversation'] = item['conversation'][:1]
            count += 1
            if len(reservoir) < n:
                reservoir.append(item)
            else:
                j = random.randint(0, count - 1)
                if j < n:
                    reservoir[j] = item
    return reservoir


def process_item(item, dataset_name, embed_db):
    first_turn = item["conversation"][0]["content"].strip()
    if not first_turn:
        return None
    if dataset_name == "wildchat":
        conversation_id = item["conversation"][0]["turn_identifier"]
    else:
        conversation_id = item["conversation_id"]
    embedding = get_embedding_with_cache(embed_db, conversation_id, first_turn)


def process_language(wildchat_dataset, lmsyschat_dataset, language):
    wildchat_embed_db = "wildchat_embeddings_cache.db"
    lmsyschat_embed_db = "lmsyschat_embeddings_cache.db"

    # ===== CACHE CHECKING LOGIC =====
    def count_cache_entries(db_name):
        """Count how many embeddings are in the cache"""
        if not os.path.exists(db_name):
            return 0
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM cache")
        count = c.fetchone()[0]
        conn.close()
        return count

    wildchat_count = count_cache_entries(wildchat_embed_db)
    lmsyschat_count = count_cache_entries(lmsyschat_embed_db)

    # Check if we can skip preprocessing
    skip_preprocessing = (
        wildchat_count >= n_per_language and lmsyschat_count >= n_per_language
    )

    if skip_preprocessing:
        print(
            f"✓ Found cached embeddings: WildChat={wildchat_count}, LMSYS={lmsyschat_count}"
        )
        print(f"  Skipping dataset loading and embedding computation")

        # Load embeddings directly from cache for UMAP training
        embeddings = []
        valid_samples = {"wildchat": [], "lmsyschat": []}

        # Load from WildChat cache
        print(f"Loading WildChat embeddings from cache...")
        conn = sqlite3.connect(wildchat_embed_db)
        c = conn.cursor()
        c.execute("SELECT key, embedding FROM cache LIMIT ?", (n_per_language,))
        for row in c.fetchall():
            conversation_id = row[0]
            embedding = json.loads(row[1])
            embeddings.append(embedding)
            # Create minimal valid sample entry
            valid_samples["wildchat"].append(
                {"conversation": [{"turn_identifier": conversation_id, "content": ""}]}
            )
        conn.close()

        # Load from LMSYS cache
        print(f"Loading LMSYS embeddings from cache...")
        conn = sqlite3.connect(lmsyschat_embed_db)
        c = conn.cursor()
        c.execute("SELECT key, embedding FROM cache LIMIT ?", (n_per_language,))
        for row in c.fetchall():
            conversation_id = row[0]
            embedding = json.loads(row[1])
            embeddings.append(embedding)
            # Create minimal valid sample entry
            valid_samples["lmsyschat"].append(
                {"conversation_id": conversation_id, "conversation": [{"content": ""}]}
            )
        conn.close()

        print(f"✓ Loaded {len(embeddings)} embeddings from cache")

        # ===== ADD THIS: NORMALIZE THE CACHED EMBEDDINGS =====
        print("Normalizing embeddings...")
        embeddings = [
            (np.array(emb) / np.linalg.norm(emb)).tolist() for emb in embeddings
        ]
        print("✓ Embeddings normalized")

    else:
        # ===== ORIGINAL PREPROCESSING CODE =====
        print(
            f"Cache incomplete (WildChat={wildchat_count}/{n_per_language}, LMSYS={lmsyschat_count}/{n_per_language})"
        )
        print("Running full preprocessing...")

        # Create databases if they don't exist
        create_database(wildchat_embed_db)
        create_database(lmsyschat_embed_db)

        random.seed(1234)
        wildchat_sampled = conditional_reservoir_sample(
            wildchat_dataset, n_per_language, language if language != "all" else None
        )
        random.seed(1234)
        lmsyschat_sampled = conditional_reservoir_sample(
            lmsyschat_dataset, n_per_language, language if language != "all" else None
        )
        random.seed(1234)

        all_items = [
            (item, "wildchat", wildchat_embed_db) for item in wildchat_sampled
        ] + [(item, "lmsyschat", lmsyschat_embed_db) for item in lmsyschat_sampled]

        print(
            f"Sampled {len(wildchat_sampled)} from WildChat and {len(lmsyschat_sampled)} from LMSYS-Chat for {language}"
        )

        # Use ThreadPoolExecutor to process items in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_item = {
                executor.submit(process_item, item, dataset, db): (item, dataset)
                for item, dataset, db in all_items
            }
            for future in tqdm(
                as_completed(future_to_item),
                total=len(all_items),
                desc="Pre-Computing embeddings",
            ):
                result = future.result()

        embeddings = []
        valid_samples = {"wildchat": [], "lmsyschat": []}

        for item in tqdm(wildchat_sampled, desc="Computing WildChat embeddings"):
            conversation_id = item["conversation"][0]["turn_identifier"]
            first_turn = item["conversation"][0]["content"].strip()
            if not first_turn:
                continue
            embedding = get_embedding_with_cache(
                wildchat_embed_db, conversation_id, first_turn
            )
            embeddings.append(embedding)
            valid_samples["wildchat"].append(item)

        for item in tqdm(lmsyschat_sampled, desc="Computing LMSYS-Chat embeddings"):
            conversation_id = item["conversation_id"]
            first_turn = item["conversation"][0]["content"].strip()
            if not first_turn:
                continue
            embedding = get_embedding_with_cache(
                lmsyschat_embed_db, conversation_id, first_turn
            )
            embeddings.append(embedding)
            valid_samples["lmsyschat"].append(item)

        print(
            f"Valid samples after filtering: WildChat: {len(valid_samples['wildchat'])}, LMSYS-Chat: {len(valid_samples['lmsyschat'])}"
        )

    print(
        f"Embedding stats: min={np.min(embeddings)}, max={np.max(embeddings)}, mean={np.mean(embeddings)}"
    )
    print(f"Any NaNs in embeddings: {np.any(np.isnan(embeddings))}")
    print(f"Any Infs in embeddings: {np.any(np.isinf(embeddings))}")

    # ===== UMAP TRAINING (RUNS FOR BOTH CACHED AND NON-CACHED) =====
    # Filter out any NaN embeddings
    valid_embeddings = []
    valid_wildchat = []
    valid_lmsyschat = []

    wildchat_end = len(valid_samples["wildchat"])
    for i, emb in enumerate(embeddings):
        if not any(np.isnan(emb)):
            valid_embeddings.append(emb)
            if i < wildchat_end:
                valid_wildchat.append(valid_samples["wildchat"][i])
            else:
                valid_lmsyschat.append(valid_samples["lmsyschat"][i - wildchat_end])

    valid_samples["wildchat"] = valid_wildchat
    valid_samples["lmsyschat"] = valid_lmsyschat
    embeddings = valid_embeddings

    print(
        f"Valid samples after filtering NaNs: WildChat: {len(valid_samples['wildchat'])}, LMSYS-Chat: {len(valid_samples['lmsyschat'])}"
    )

    # # SUBSAMPLE for UMAP training only
    # print(f"Subsampling {len(embeddings)} embeddings to 10,000 for UMAP training...")
    # random.seed(1234)
    # indices = random.sample(range(len(embeddings)), min(10000, len(embeddings)))
    # embeddings_subsample = [embeddings[i] for i in indices]

    # Train UMAP with memory-friendly parameters
    print("Training UMAP model...")
    umap = ParametricUMAP(
        n_components=2,
        n_neighbors=50,
        spread=0.2,
        min_dist=0.1,
        metric="cosine",
        verbose=True,
    )
    umap.fit(embeddings)

    # But apply to ALL samples after training
    print("Applying trained UMAP to all embeddings...")

    # Save UMAP model
    os.makedirs(f"umap_model/{language}", exist_ok=True)
    if hasattr(umap, "_raw_data"):
        del umap._raw_data
    if hasattr(umap, "knn_search_index") and hasattr(
        umap.knn_search_index, "_raw_data"
    ):
        del umap.knn_search_index._raw_data
    umap.save(f"umap_model/{language}")

    for model_path in [
        f"umap_model/{language}/model.pkl",
        f"umap_model/{language}/parametric_model.keras",
        f"umap_model/{language}/scaler.pkl",
    ]:
        if os.path.exists(model_path):
            print(f"Removing {model_path}")
            os.remove(model_path)

    umap_encoder = keras.models.load_model(
        os.path.join(f"umap_model/{language}", "encoder.keras")
    )

    for db_path in [
        f"umap_{language}_wildchat_cache.db",
        f"umap_{language}_lmsyschat_cache.db",
    ]:
        if os.path.exists(db_path):
            print(f"Removing {db_path}")
            os.remove(db_path)

    wildchat_umap_db = create_database(f"umap_{language}_wildchat_cache.db")
    lmsyschat_umap_db = create_database(f"umap_{language}_lmsyschat_cache.db")

    # Apply UMAP to all samples and save results
    for dataset_name in ["wildchat", "lmsyschat"]:
        json_data = []
        start_index = (
            0 if dataset_name == "wildchat" else len(valid_samples["wildchat"])
        )

        for i, item in enumerate(valid_samples[dataset_name], start=start_index):
            conversation_id = (
                item["conversation"][0]["turn_identifier"]
                if dataset_name == "wildchat"
                else item["conversation_id"]
            )
            umap_embedding = umap_encoder(np.array([embeddings[i]])).numpy()[0].tolist()

            if dataset_name == "wildchat":
                insert_or_update(wildchat_umap_db, conversation_id, "", umap_embedding)
            else:
                insert_or_update(lmsyschat_umap_db, conversation_id, "", umap_embedding)

            json_data.append(
                {
                    "i": conversation_id,
                    "e": [
                        round(float(umap_embedding[0]), 4),
                        round(float(umap_embedding[1]), 4),
                    ],
                    "c": item["conversation"][0]["content"],
                    "d": dataset_name,
                }
            )

        os.makedirs(f"static/{language}", exist_ok=True)
        with open(f"static/{language}/{dataset_name}_embeddings_all.json", "w") as f:
            json.dump(json_data, f)

        subsampled_json_data = random.sample(json_data, min(1500, len(json_data)))
        subsampled_json_path = f"static/{language}/{dataset_name}_embeddings.json"
        with open(subsampled_json_path, "w") as f:
            json.dump(subsampled_json_data, f)
        gzip_file(subsampled_json_path, f"{subsampled_json_path}.gz")


# Main processing loop
print("Loading datasets...")
wildchat_dataset = load_dataset("allenai/WildChat-1M-Full")['train']
lmsyschat_dataset = load_dataset("lmsys/LMSYS-Chat-1M")['train']

for language in LANGUAGES:
    print(f"Processing {language}...")
    process_language(wildchat_dataset, lmsyschat_dataset, language)

print("Processing complete!")
