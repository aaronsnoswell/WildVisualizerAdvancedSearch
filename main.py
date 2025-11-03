# pylint: disable=global-statement,redefined-outer-name
import argparse
import re
import csv
import glob
import keras
import json
import random
import os
from markupsafe import Markup, escape
import sqlite3
from openai import OpenAI
import numpy as np
import joblib
#from umap.parametric_umap import load_ParametricUMAP
import tiktoken
#from sklearn.decomposition import PCA
from umap.parametric_umap import ParametricUMAP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


indices = ["wildchat", "lmsyschat"]
supported_fields = {
    "wildchat": [
        "dataset",
        "toxic",
        "redacted",
        "model",
        "hashed_ip",
        "language",
        "country",
        "state",
        "min_turns",
        "conversation_id",
    ],
    "lmsyschat": [
        "dataset",
        "toxic",
        "redacted",
        "model",
        "language",
        "min_turns",
        "conversation_id",
    ],
}


def parse_search_query(query_string):
    """
    Parse search query string into structured format.

    Syntax: contains:"text"[&turn:(user|assistant)] nocontains:"text"[&turn:(user|assistant)]

    Returns:
        List of dicts: [
            {'type': 'contains', 'text': 'Hello', 'turn_role': 'user'},
            {'type': 'nocontains', 'text': 'python', 'turn_role': None}
        ]
    """
    if not query_string or not query_string.strip():
        return []

    pattern = r'(contains|nocontains):"([^"]+)"(?:&turn:(user|assistant))?'

    parsed = []
    for match in re.finditer(pattern, query_string, re.IGNORECASE):
        query_type = match.group(1).lower()
        text = match.group(2)
        turn_role = match.group(3).lower() if match.group(3) else None

        parsed.append({"type": query_type, "text": text, "turn_role": turn_role})

    if not parsed:
        parsed.append(
            {"type": "contains", "text": query_string.strip(), "turn_role": None}
        )

    return parsed


def build_pagination_url(page_num, request_args):
    """
    Build a pagination URL that preserves all current query parameters.

    Args:
        page_num: The page number to link to
        request_args: The request.args object from Flask

    Returns:
        String URL with all parameters preserved
    """
    params = {}

    # Copy all existing parameters except 'page'
    for key in request_args:
        if key != "page":
            params[key] = request_args[key]

    # Add the new page number
    params["page"] = page_num

    # Build URL
    return "?" + urlencode(params)


def build_content_query(query_item):
    """
    Build nested Elasticsearch query for a single contains/nocontains item.

    Args:
        query_item: Dict with 'type', 'text', and optional 'turn_role'

    Returns:
        Elasticsearch nested query dict
    """
    must_clauses = [{"match_phrase": {"conversation.content": query_item["text"]}}]

    # Add turn role constraint if specified
    if query_item["turn_role"]:
        must_clauses.append({"term": {"conversation.role": query_item["turn_role"]}})

    return {
        "nested": {"path": "conversation", "query": {"bool": {"must": must_clauses}}}
    }


def validate_search_query(query_string):
    """
    Validate search query syntax and return error message if invalid.

    Returns:
        None if valid, error message string if invalid
    """
    if not query_string or not query_string.strip():
        return None

    # Check for unmatched quotes
    if query_string.count('"') % 2 != 0:
        return "Unmatched quotes in query"

    # Check if any valid patterns found
    pattern = r'(contains|nocontains):"([^"]+)"(?:&turn:(user|assistant))?'
    matches = list(re.finditer(pattern, query_string, re.IGNORECASE))

    if not matches:
        return None  # Plain text is valid - will be treated like a contains: search for backward compatability

    # Check for empty search text in structured queries
    for match in matches:
        if not match.group(2).strip():
            return "Empty search text not allowed"

    return None


def build_query_for_index(index_name, filters, search_query, from_, size_):
    """
    Build Elasticsearch query combining search_query and filters.

    Args:
        index_name: 'wildchat' or 'lmsyschat'
        filters: Dict of metadata filters (toxic, language, etc.)
        search_query: String with advanced query syntax
        from_: Pagination start
        size_: Number of results

    Returns:
        Tuple: (query_dict, any_filters_applied)
    """
    # Parse search query
    parsed_queries = parse_search_query(search_query)

    must_clauses = []
    must_not_clauses = []

    # Build content queries from parsed search
    for q in parsed_queries:
        content_query = build_content_query(q)

        if q["type"] == "contains":
            must_clauses.append(content_query)
        elif q["type"] == "nocontains":
            must_not_clauses.append(content_query)

    # Add existing filter logic
    if filters["toxic"]:
        if index_name == "wildchat":
            must_clauses.append({"term": {"toxic": filters["toxic"] == "true"}})
        else:
            must_clauses.append(
                {
                    "nested": {
                        "path": "openai_moderation",
                        "query": {
                            "term": {
                                "openai_moderation.flagged": filters["toxic"] == "true"
                            }
                        },
                    }
                }
            )

    if filters["redacted"]:
        must_clauses.append({"term": {"redacted": filters["redacted"] == "true"}})

    if filters["model"]:
        must_clauses.append({"term": {"model": filters["model"]}})

    if filters["hashed_ip"]:
        must_clauses.append({"term": {"hashed_ip": filters["hashed_ip"]}})

    if filters["language"]:
        must_clauses.append({"term": {"language": filters["language"]}})

    if filters["country"]:
        must_clauses.append({"term": {"country": filters["country"]}})

    if filters["state"] and index_name == "wildchat":
        must_clauses.append({"term": {"state": filters["state"]}})

    # Note: ElasticSearch based min_turns filtering is disabled due to conflict with nested content queries
    # Filtering is done post-query in Python instead
    # if filters['min_turns']:
    #     must_clauses.append({
    #         "script": {
    #             "script": {
    #                 "source": "params._source.conversation.size() >= params.min_turns",
    #                 "params": {"min_turns": int(filters['min_turns'])}
    #             }
    #         }
    #     })

    if filters["conversation_id"]:
        if index_name == "wildchat":
            must_clauses.append(
                {
                    "nested": {
                        "path": "conversation",
                        "query": {
                            "term": {
                                "conversation.turn_identifier": filters[
                                    "conversation_id"
                                ]
                            }
                        },
                    }
                }
            )
        else:
            must_clauses.append(
                {"term": {"conversation_id": filters["conversation_id"]}}
            )

    # Check if any filters applied
    any_filters = bool(must_clauses or must_not_clauses)

    # Build final query
    if any_filters:
        bool_query = {}
        if must_clauses:
            bool_query["must"] = must_clauses
        if must_not_clauses:
            bool_query["must_not"] = must_not_clauses

        query = {"query": {"bool": bool_query}, "from": from_, "size": size_}
    else:
        query = {"query": {"match_all": {}}, "from": from_, "size": size_}

    return query, any_filters


def nl2br(value):
    escaped_value = escape(value)
    return Markup(escaped_value.replace('\n', Markup('<br>')))

import yaml
from flask import Flask, jsonify, redirect, render_template, send_from_directory, request, url_for, send_file, abort, after_this_request
from flask_frozen import Freezer
from flaskext.markdown import Markdown
from elasticsearch import Elasticsearch, helpers

site_data = {}
by_uid = {}

es = Elasticsearch('https://localhost:9200', basic_auth=('elastic', os.getenv('ES_PASSWD')), ssl_assert_fingerprint=os.getenv('ES_FINGERPRINT'))

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
tokenizer = tiktoken.get_encoding('cl100k_base')

# Load the PCA model

embedding_projectors = {}
for folder in glob.glob(os.path.join('umap_model', '*')):
   #scaler_path = os.path.join(folder, 'scaler.pkl')
   umap_path = folder
   #if os.path.exists(scaler_path):
   if os.path.exists(umap_path):
       language = os.path.basename(folder)
       #scaler = joblib.load(scaler_path)
       try:
           umap_encoder = keras.models.load_model(os.path.join(umap_path, "encoder.keras"))
           #umap = load_ParametricUMAP(umap_path)
           #embedding_projectors[language] = {'scaler': scaler, 'umap': umap}
           embedding_projectors[language] = umap_encoder
       except Exception as e:
           print (e)
print (embedding_projectors.keys())

def create_database(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cache
                 (key TEXT PRIMARY KEY, prompt TEXT, embedding TEXT)''')
    conn.commit()
    conn.close()

#create_database('embeddings_cache.db')
#create_database('umap_cache.db')

def insert_or_update(db_name, key, prompt, embedding):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO cache
                 (key, prompt, embedding) VALUES (?, ?, ?)''', 
                 (key, prompt, json.dumps(embedding)))
    conn.commit()
    conn.close()

def retrieve(db_name, key):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("SELECT embedding FROM cache WHERE key=?", (key,))
    result = c.fetchone()
    conn.close()
    if result:
        return True, json.loads(result[0])
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


def main(site_data_path):
    global site_data, extra_files
    extra_files = ["README.md"]
    # Load all for your sitedata one time.
    for f in glob.glob(site_data_path + "/*"):
        extra_files.append(f)
        try:
            name, typ = f.split("/")[-1].split(".")
        except Exception as e:
            continue
        if typ == "json":
            site_data[name] = json.load(open(f))
        elif typ in {"csv", "tsv"}:
            site_data[name] = list(csv.DictReader(open(f)))
        elif typ == "yml":
            site_data[name] = yaml.load(open(f).read(), Loader=yaml.SafeLoader)

    for typ in ["papers", "speakers", "workshops"]:
        by_uid[typ] = {}
        for p in site_data[typ]:
            by_uid[typ][p["UID"]] = p

    print("Data Successfully Loaded")
    return extra_files

extra_files = main('sitedata')
# ------------- SERVER CODE -------------------->

app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.do')
app.jinja_env.filters['nl2br'] = nl2br
app.config.from_object(__name__)
freezer = Freezer(app)
markdown = Markdown(app)


# MAIN PAGES
def _data():
    data = {}
    data["config"] = site_data["config"]
    return data

@app.route("/favicon.ico")
def favicon():
    return send_from_directory('sitedata', "favicon.ico")

# TOP LEVEL PAGES
@app.route("/")
def index():
    data = _data()
    search_query = request.args.get("search_query", "")  # CHANGED from 'contains'
    page = int(request.args.get("page", 1))

    # Construct the Elasticsearch query
    filters = {
        "dataset": request.args.get("dataset", ""),
        "toxic": request.args.get("toxic", ""),
        "redacted": request.args.get("redacted", ""),
        "model": request.args.get("model", ""),
        "hashed_ip": request.args.get("hashed_ip", ""),
        "language": request.args.get("language", ""),
        "country": request.args.get("country", ""),
        "state": request.args.get("state", ""),
        "min_turns": request.args.get("min_turns", ""),
        "conversation_id": request.args.get("conversation_id", ""),
    }

    # Validate search query
    error = validate_search_query(search_query)
    if error:
        data["error"] = error
        data["search_query"] = search_query
        return render_template("index.html", **data)

    disabled_datasets = []
    for dataset in indices:
        for field in filters:
            if filters[field]:
                if field not in supported_fields[dataset]:
                    disabled_datasets.append(dataset)
    indices_to_search = []
    if (
        filters["dataset"] == "" or filters["dataset"] == "wildchat"
    ) and "wildchat" not in disabled_datasets:
        indices_to_search.append("wildchat")
    if (
        filters["dataset"] == "" or filters["dataset"] == "lmsyschat"
    ) and "lmsyschat" not in disabled_datasets:
        indices_to_search.append("lmsyschat")
    size = max(30 // len(indices_to_search), 1)
    from_ = (page - 1) * size
    if from_ >= 10000:
        return render_template(
            "error.html",
            message="You cannot navigate beyond the 10,000th result. Please refine your search by going to earlier pages.",
        )
    if from_ + size > 10000:
        size_ = 10000 - from_
    else:
        size_ = size

    any_filters = False
    if "dataset" in filters and filters["dataset"] != "":
        any_filters = True
    conversations = []
    total = 0
    assert len(indices_to_search) > 0
    for index_name in indices_to_search:
        # Execute search query
        search_query_obj, any_filters_ = build_query_for_index(
            index_name,
            filters,
            search_query,
            from_,
            size_,
        )
        any_filters = any_filters or any_filters_
        response = es.search(index=index_name, body=search_query_obj)
        conversations_raw = [hit["_source"] for hit in response["hits"]["hits"]]

        for conversation_raw in conversations_raw:
            conversation = {}
            conversation["dataset"] = index_name
            for key in [
                "timestamp",
                "country",
                "state",
                "hashed_ip",
                "model",
                "toxic",
                "redacted",
                "conversation",
                "conversation_id",
            ]:
                if key in conversation_raw:
                    conversation[key] = conversation_raw[key]

            # Min turn filtering happens post-query in Python
            if filters["min_turns"]:
                min_turns_required = int(filters["min_turns"])
                if len(conversation_raw.get("conversation", [])) < min_turns_required:
                    continue  # Skip this conversation - min turns too small

            if index_name == "wildchat":
                conversation["conversation_id"] = conversation_raw["conversation"][0][
                    "turn_identifier"
                ]
            if index_name == "lmsyschat":
                conversation["toxic"] = any(
                    [item["flagged"] for item in conversation_raw["openai_moderation"]]
                )
            conversations.append(conversation)
        total = max(total, response["hits"]["total"]["value"])
    # total_pages = (total // size) + 1
    total_pages = (total + size - 1) // size
    random.seed(1234)
    random.shuffle(conversations)

    # Pagination logic
    pages = []
    if total_pages > 1:
        if page > 3:
            pages.append(1)
            if page > 4:
                pages.append("...")
        pages.extend(range(max(1, page - 2), min(total_pages + 1, page + 3)))
        if page < total_pages - 3:
            if page < total_pages - 4:
                pages.append("...")
            pages.append(total_pages)
    # import pdb; pdb.set_trace()
    data.update(
        {
            "conversations": conversations,
            "search_query": search_query,
            "page": page,
            "pages": pages,
            "total": total,
            "filters": filters,
            "any_filters": any_filters,
            "build_pagination_url": lambda p: build_pagination_url(p, request.args),
        }
    )
    return render_template("index.html", **data)


@app.route("/search_embeddings", methods=["POST"])
def search_embeddings():
    filters = request.json
    search_expansion_limit = filters["search_expansion_limit"]
    del filters["search_expansion_limit"]
    if search_expansion_limit == "":
        search_expansion_limit = "100"
    search_expansion_limit = int(search_expansion_limit)
    search_expansion_limit = max(0, min(search_expansion_limit, 2000))

    search_query = filters["search_query"]
    del filters["search_query"]
    visualization_language = filters["visualization_language"]
    del filters["visualization_language"]

    umap_encoder = embedding_projectors[visualization_language]

    disabled_datasets = []
    for dataset in indices:
        for field in filters:
            if filters[field]:
                if field not in supported_fields[dataset]:
                    disabled_datasets.append(dataset)
    indices_to_search = []
    if (
        filters["dataset"] == "" or filters["dataset"] == "wildchat"
    ) and "wildchat" not in disabled_datasets:
        indices_to_search.append("wildchat")
    if (
        filters["dataset"] == "" or filters["dataset"] == "lmsyschat"
    ) and "lmsyschat" not in disabled_datasets:
        indices_to_search.append("lmsyschat")
    any_filters = False

    for index_name in indices_to_search:
        # Execute search query
        search_query_obj, any_filters_ = build_query_for_index(
            index_name, filters, search_query, 0, 10000
        )
        any_filters = any_filters or any_filters_

    conversations = []
    if any_filters:
        if (("language" not in filters) or (not filters["language"])) and (
            visualization_language != "all"
        ):
            filters["language"] = visualization_language
        conversation_ids = set([])
        for index_name in indices_to_search:
            search_query_obj, any_filters_ = build_query_for_index(
                index_name, filters, search_query, 0, search_expansion_limit
            )
            response = es.search(index=f"{index_name}_subset", body=search_query_obj)
            if response["hits"]["total"]["value"] < 30:
                response = es.search(index=index_name, body=search_query_obj)
            conversations_raw = [hit["_source"] for hit in response["hits"]["hits"]]
            for conversation_raw in conversations_raw:
                conversation = {}
                conversation["dataset"] = index_name
                for key in [
                    "timestamp",
                    "country",
                    "state",
                    "hashed_ip",
                    "model",
                    "toxic",
                    "redacted",
                    "conversation",
                    "conversation_id",
                ]:
                    if key in conversation_raw:
                        conversation[key] = conversation_raw[key]
                if index_name == "lmsyschat":
                    conversation["toxic"] = any(
                        [
                            item["flagged"]
                            for item in conversation_raw["openai_moderation"]
                        ]
                    )
                if index_name == "wildchat":
                    conversation["conversation_id"] = conversation_raw["conversation"][
                        0
                    ]["turn_identifier"]
                conversation_id = conversation["conversation_id"]
                if conversation_id not in conversation_ids:
                    conversations.append(conversation)
                    conversation_ids.add(conversation_id)

    conversation_embeddings = {}
    print("#Matched Conversation:", len(conversations))
    for conversation in conversations:
        dataset = conversation["dataset"]
        conversation_id = conversation["conversation_id"]
        umap_database_name = f"umap_{visualization_language}_{dataset}_cache.db"
        embed_database_name = f"{dataset}_embeddings_cache.db"

        hit, embedding_2d = retrieve(umap_database_name, conversation_id)
        if not hit:
            print("not hit")
            conversation_text = conversation["conversation"][0]["content"]
            conversation_text = conversation_text.strip()
            if not conversation_text:
                continue
            embedding = get_embedding_with_cache(
                embed_database_name,
                conversation_id,
                conversation_text,
                model="text-embedding-3-small",
            )
            embedding_2d = umap_encoder(np.array([embedding])).numpy()[0]
            insert_or_update(
                umap_database_name,
                conversation_id,
                "",
                [float(embedding_2d[0]), float(embedding_2d[1])],
            )
        conversation_embeddings[str(conversation_id)] = {
            "i": conversation_id,
            "e": [round(float(embedding_2d[0]), 4), round(float(embedding_2d[1]), 4)],
            "c": conversation["conversation"][0]["content"],
            "d": dataset,
        }
    return jsonify(conversation_embeddings)


@app.route("/embeddings/<language>")
@app.route("/embeddings")
def embeddings(language=None):
    data = _data()

    search_query = request.args.get("search_query", "")
    # Construct the Elasticsearch query
    filters = {
        "toxic": request.args.get("toxic", ""),
        "redacted": request.args.get("redacted", ""),
        "model": request.args.get("model", ""),
        "hashed_ip": request.args.get("hashed_ip", ""),
        "language": request.args.get("language", ""),
        "country": request.args.get("country", ""),
        "state": request.args.get("state", ""),
        "min_turns": request.args.get("min_turns", ""),
        "search_expansion_limit": request.args.get("search_expansion_limit", ""),
        "conversation_id": request.args.get("conversation_id", ""),
    }

    any_filters = False
    for key in filters:
        if filters[key]:
            any_filters = True
    if search_query:
        any_filters = True

    data.update(
        {
            "search_query": search_query,
            "filters": filters,
            "any_filters": any_filters,
            "visualization_language": language or "all",
        }
    )
    return render_template("embeddings.html", **data)

def extract_list_field(v, key):
    value = v.get(key, "")
    if isinstance(value, list):
        return value
    else:
        return value.split("|")

@app.route("/conversation/wildchat/<int:turn_identifier>")
def conversation_wildchat(turn_identifier):
    data = _data()
    search_query = {
        "query": {
            "nested": {
                "path": "conversation",
                "query": {
                    "term": {
                        "conversation.turn_identifier": turn_identifier
                    }
                }
            }
        }
    }

    response = es.search(index="wildchat", body=search_query)
    if not response['hits']['hits']:
        return render_template("error.html", message="Conversation not found."), 404

    # Extract the conversation and check the first turn identifier
    conversation = response['hits']['hits'][0]['_source']
    first_turn_identifier = conversation['conversation'][0]['turn_identifier']

    # If the turn_identifier is not the first turn, redirect to the first turn's page
    if turn_identifier != first_turn_identifier:
        return redirect(url_for('conversation_wildchat', turn_identifier=first_turn_identifier))

    conversation['conversation_id'] = first_turn_identifier
    conversation['dataset'] = 'wildchat'
    data["conversation"] = conversation
    data["from_page"] = request.args.get('from', 'filter')
    data["visualization_language"] = request.args.get('lang', 'all')
    return render_template("conversation.html", **data)

@app.route("/conversation/lmsyschat/<string:conversation_id>")
def conversation_lmsyschat(conversation_id):
    data = _data()
    search_query = {
        "query": {
            "term": {"conversation_id": conversation_id}
        }
    }

    response = es.search(index="lmsyschat", body=search_query)
    if not response['hits']['hits']:
        return render_template("error.html", message="Conversation not found."), 404

    # Extract the conversation and check the first turn identifier
    conversation = response['hits']['hits'][0]['_source']
    conversation['dataset'] = 'lmsyschat'
    conversation['toxic'] = any([item['flagged'] for item in conversation['openai_moderation']])
    data["conversation"] = conversation
    data["from_page"] = request.args.get('from', 'filter')
    data["visualization_language"] = request.args.get('lang', 'all')
    return render_template("conversation.html", **data)



if __name__ == "__main__":
    debug_val = False
    if os.getenv("FLASK_DEBUG") == "True":
        debug_val = True

    app.run(port=8080, debug=debug_val, extra_files=extra_files, host='0.0.0.0')
