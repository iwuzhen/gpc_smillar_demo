import streamlit as st
import neo4j
import pickle
from elasticsearch import Elasticsearch
import collections
import pandas as pd
import concurrent.futures
from functools import lru_cache
from modelscope import AutoModel
from numpy.linalg import norm
import pathlib
import requests
import redis
import psycopg
import json
import gzip


@st.cache_resource
def get_redis_pool():
    connection_pool = redis.ConnectionPool(host='192.168.1.227', port=6379, db=0)
    # 创建 Redis 客户端实例，使用上面创建的连接池
    REDIS = redis.Redis(connection_pool=connection_pool)
    return REDIS


@st.cache_resource
def get_elastic_connection():
    """
    获取Elasticsearch连接。

    此函数从secrets字典中获取存储的elastic_uri，然后返回一个新的Elasticsearch连接实例。
    
    :return: Elasticsearch: 一个Elasticsearch客户端连接实例，用于与Elasticsearch集群通信。
    """
    return Elasticsearch(st.secrets["elastic_uri"])


@st.cache_resource
# @st.cache_data(max_entries=1_000)
def get_token_data():
    current_path =pathlib.Path(__file__).joinpath("../../model/WTD.pkl.gz").resolve()
    with gzip.open(current_path, 'rb') as f:
        WTD = pickle.load(f)
    return WTD


@st.cache_resource
def get_neo4j_connection():
    """
    初始化并返回一个Neo4j数据库的异步驱动程序。

    该函数通过使用Secrets Manager获取Neo4j数据库的URI、用户名和密码，
    来建立一个安全的、异步的数据库连接。

    返回:
        neo4j.AsyncGraphDatabase.driver: 一个配置好的Neo4j异步驱动程序实例，
                                          用于进行异步数据库操作。
    """
    # 使用Secrets Manager中的凭证建立Neo4j数据库连接
    return neo4j.GraphDatabase.driver(
            st.secrets["neo4j_uri"],
            auth=(st.secrets["neo4j_user"], st.secrets["neo4j_password"],),
        )


@st.cache_resource
def get_jinaai_mode():
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method
    return model

@st.cache_resource
def get_sts_model():
    JinNaAI_Model = get_jinaai_mode()
    return JinNaAI_Model

cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))

def calculate_similarity(sentence1, sentence2):
    JinNaAI_Model = get_sts_model()
    embeddings = JinNaAI_Model.encode([sentence1, sentence2])
    return cos_sim(embeddings[0], embeddings[1])

def format_text(text):
    return text.split(" ")


WTD = get_token_data()
def get_token(text):
    stop_word = set(["is","they","their","in", "of", "are","it","as","thy","the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "what", "why", "how", "where", "when", "which", "who", "whom", "whose", "which", "that", "this", "these", "those", "such", "very",])
    token_set = set()
    annotated_text = []
    
    token_list = format_text(text)
    token_range = 5
    start_token_index = 0
    end_token_index = token_range
    last_token_index = 0
    
    while start_token_index < len(token_list):
        if token_list[start_token_index].lower() in stop_word:
            start_token_index += 1
            end_token_index = start_token_index + token_range
            continue
        
        sub_token_list = token_list[start_token_index:end_token_index]
        sub_text = " ".join(sub_token_list).strip()
        ID = WTD.get(sub_text.lower().strip(".:, (){}*").strip())
        
        if ID:
            # 前置空白区
            last_string = " ".join(token_list[last_token_index:start_token_index])
            if last_string:
                annotated_text.append(last_string)
                
            # 当前 token 区
            annotated_text.append((sub_text, str(ID)))
            last_token_index = end_token_index
            start_token_index = end_token_index
            end_token_index = start_token_index + token_range
            
            token_set.add((ID, sub_text.strip(".:, (){}*").strip()))
        else:
            if end_token_index - start_token_index > 1: 
                end_token_index -= 1
            else:
                start_token_index += 1
                end_token_index = start_token_index + token_range
            
        
    last_string = " ".join(token_list[last_token_index:start_token_index])
    if last_string:
        annotated_text.append(last_string)
    return token_set, annotated_text


def get_plaintext(pageID):
    ES = get_elastic_connection()
    response = ES.search(
        index="en_page",
        body={
            "_source": ["title", "id", "plaintext"],
            "query": {
                "match_phrase": {
                    "_id": pageID,
                },
            },
            "size": 1,
        },
    )
    # print(pageID)
    if response["hits"]["hits"]:
        # print(response["hits"]["hits"][0])
        text_split  = response["hits"]["hits"][0]['_source'].get("plaintext","").strip().split("\n")
        if text_split:
            plain_text = ""
            for text in text_split:
                plain_text += f"{text} " 
                if len(plain_text.strip().split(" ")) > 15:
                    return plain_text.strip()
            return plain_text
    return None


@lru_cache(maxsize=1000)
# function with list of labels
def search_wikipedia(searchterm: str) -> list[any]:
    ES = get_elastic_connection()
    query = searchterm
    if not query:
        return []
    response = ES.search(
            index="wikipedia_title",
            body={
                "_source": ["title", "id", "redirect"],
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "title.standard": {"query": query, "boost": 100},
                                },
                            },
                            {
                                "match": {
                                    "redirect.standard": {"query": query, "boost": 5},
                                },
                            },
                            {"match": {"title.prefix": {"query": query, "boost": 20}}},
                            {
                                "match": {
                                    "redirect.prefix": {"query": query, "boost": 4},
                                },
                            },
                            {
                                "match": {
                                    "title.standard": {"query": query, "fuzziness": 2},
                                },
                            },
                            {
                                "match": {
                                    "redirect.standard": {
                                        "query": query,
                                        "fuzziness": 2,
                                    },
                                },
                            },
                        ],
                    },
                },
                "highlight": {
                    "fragment_size": 40,
                    "fields": {
                        "title": {},
                    },
                },
                "size": 15,
            },
        )
    ret_list = []
    for hit in response["hits"]["hits"]:
        doc = hit["_source"]
        doc["id"] = int(hit["_id"])
        doc["highlight"] = hit.get("highlight", {})

        doc["token"] = doc["title"]
        del doc["title"]
        ret_list.append(doc)
    return [(item['token'], item['id']) for item in ret_list] if ret_list else []


@lru_cache(maxsize=1000)
def query_single_distancles(ID, ID_tuple):
    # REDIS
    redis_drive = get_redis_pool()
    key_list = []
    for item in ID_tuple:
        if ID > item:
            key_list.append(f"{ID}-{item}")
        else:
            key_list.append(f"{item}-{ID}")
            
    redis_cache_result  = redis_drive.mget(key_list)
    new_ID_list = []
    
    record_list = []
    for i, value in enumerate(redis_cache_result):
        if value: 
            record_list.append((ID, ID_tuple[i], float(value)))
        else:
            new_ID_list.append(ID_tuple[i])
        
        
    mset_data = {}
    driver = get_neo4j_connection()
    with driver.session(database="neo4j") as session:
        result = session.run(
            "MATCH (start:P {Id: $source})<-[r:D]->(end:P) WHERE end.Id IN $target AND r.weight < 1 "
            "WITH start.Id as a, end.Id as b, r.weight as w "
            "ORDER BY w ASC "
            # "LIMIT 1 "
            "RETURN a, b, w "
            ,
            source=ID,
            target=new_ID_list,
        )
        for record in result:
            # record_list.append(record)
            record_list.append((record[0],record[1],record[2]))
            
            if record[0] > record[1]:
                mset_data[f"{record[0]}-{record[1]}"] = record[2]
            else:
                mset_data[f"{record[1]}-{record[0]}"] = record[2]
    

    if mset_data: 
        redis_drive.mset(mset_data)
        # record = result.single()
    return record_list


@lru_cache(maxsize=1000)
def query_partner_distancles(ID_a, ID_b):
    if ID_a == 0 or ID_b == 0:
        return 0
    if ID_a == ID_b:
        return 1
    driver = get_neo4j_connection()
    with driver.session(database="neo4j") as session:
        result = session.run(
            "MATCH (start:P {Id: $source})<-[r:D]->(end:P {Id: $dest}) "
            "WITH r.weight as w "
            "LIMIT 1 "
            "RETURN w "
            ,
            source=ID_a,
            dest=ID_b,
        )
        record = result.single()
    if record:
        return 1-record['w']
    return 0


def calculate_cartesian_product_similarity(tuple_1, tuple_2, interest_tuple):
    # 查询得到优先权重小的边有效边,距离大的边标记为无效边
    name_dict = {}
    for ID, name in tuple_1 + tuple_2:
        name_dict[int(ID)] = name
    
    ID_tuple_2 =  tuple([int(item[0]) for item in tuple_2])
    
    dup_map_a = collections.defaultdict(list)
    dup_map_b = collections.defaultdict(list)
    # for item in tuple_1:
    #     ID = int(item[0])
    #     result = query_single_distancles(ID, ID_tuple_2)
    #     if result:
    #         dup_map[result[1]].append((result[2], result[0], result[1]))
            
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(query_single_distancles, int(item[0]), ID_tuple_2, ) for item in tuple_1}

    # 双向保留
    for future in concurrent.futures.as_completed(futures):
        for result in future.result():
            dup_map_a[result[1]].append((result[2], result[0], result[1]))
            dup_map_b[result[0]].append((result[2], result[0], result[1]))
        
    
    doc_recored_remove = []
    doc_recored_ok = []
    for item in interest_tuple:
        title = item[1]
        doc_recored_ok.append(
            {
            'token a': title,
            'token b': title,
            'similarity': 1,
            }
        )
    
    # 双向过滤
    ok_set = set()
    fail_set = set()
    for dup_map in [dup_map_a,dup_map_b]:
        for items in dup_map.values():
            items.sort(key=lambda x: x[0])
            ok_set.add(items[0])
            if len(items) > 1:
                fail_set.update(items[1:])
    # print(len(ok_set),len(fail_set))
    for item in ok_set:
        doc_recored_ok.append({
            'token a': name_dict[item[1]],
            'token b': name_dict[item[2]],
            'similarity': 1-item[0],
        })
    for item in fail_set - ok_set:
        doc_recored_remove.append({
            'token a': name_dict[item[1]],
            'token b': name_dict[item[2]],
            'similarity': 1-item[0],
        })

    df_ok = pd.DataFrame(doc_recored_ok)
    df_out = pd.DataFrame(doc_recored_remove)
    return df_ok, df_out



def get_llm_article(title):
    if title == "":
        return ""

    url = "https://api.siliconflow.cn/v1/chat/completions"

    payload = {
        "model": "Qwen/Qwen2-7B-Instruct",
        "messages": [
            {
                "role": "user",
                "content":  "# You are an encyclopedia writer. \n"
                            f"Please write an article for *{title}* that is easy to understand and should be around 50 words long in English"
            }
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {st.secrets['LLM_SECRET']}"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print(response.json())
            return response.json()['choices'][0]['message']['content']
        else:
            print(response.text)
    except Exception as e:
        print(e)
        
    return "sorry! something went wrong"


@lru_cache(maxsize=1000)
def query_distance(tuple_set_1, tuple_set_2):
    pg_uri = st.secrets['postgres_uri']
    ret = []
    with psycopg.connect(pg_uri) as conn:
        with conn.cursor() as cursor:
            try:
                for x in tuple_set_1:
                    greater_than_or_equal_x = list(item for item in tuple_set_2 if item > x)
                    if greater_than_or_equal_x:
                        cursor.execute("SELECT source_id,target_id,weight FROM edges WHERE source_id = %s AND target_id = ANY(%s);", 
                                    (x, greater_than_or_equal_x,)
                                    ) 
                        for result in cursor:
                            ret.append(result)
                if tuple_set_1 != tuple_set_2:
                    for x in tuple_set_2:
                        greater_than_or_equal_x = list(item for item in tuple_set_1 if item > x)
                        if greater_than_or_equal_x:
                            cursor.execute("SELECT source_id,target_id,weight FROM edges WHERE source_id = %s AND target_id = ANY(%s);", 
                                        (x, greater_than_or_equal_x,)
                                        ) 
                            for result in cursor:
                                ret.append(result)
            except TypeError as e:
                print(tuple_set_1)
                print(tuple_set_2)
                raise e
    return ret


def calculate_cartesian_product_similarity_pgs(tuple_1, tuple_2, interest_tuple):
    name_dict = {}
    for ID, name in tuple_1 + tuple_2:
        name_dict[int(ID)] = name
    
    ID_tuple_2 =  tuple([int(item[0]) for item in tuple_2])
    ID_tuple_1 =  tuple([int(item[0]) for item in tuple_1])
    
    dup_map_a = collections.defaultdict(list)
    dup_map_b = collections.defaultdict(list)
    
    for result in query_distance(ID_tuple_1,ID_tuple_2):
        dup_map_a[result[1]].append((result[2], result[0], result[1]))
        dup_map_b[result[0]].append((result[2], result[0], result[1]))
    
    
    doc_recored_remove = []
    doc_recored_ok = []
    for item in interest_tuple:
        title = item[1]
        doc_recored_ok.append(
            {
            'token a': title,
            'token b': title,
            'similarity': 1,
            }
        )
    
    # 双向过滤
    ok_set = set()
    fail_set = set()
    for dup_map in [dup_map_a,dup_map_b]:
        for items in dup_map.values():
            items.sort(key=lambda x: x[0])
            ok_set.add(items[0])
            if len(items) > 1:
                fail_set.update(items[1:])
    # print(len(ok_set),len(fail_set))
    for item in ok_set:
        doc_recored_ok.append({
            'token a': name_dict[item[1]],
            'token b': name_dict[item[2]],
            'similarity': max(1-item[0],0 ),
        })
    for item in fail_set - ok_set:
        doc_recored_remove.append({
            'token a': name_dict[item[1]],
            'token b': name_dict[item[2]],
            'similarity': max(1-item[0],0 ),
        })

    df_ok = pd.DataFrame(doc_recored_ok)
    df_out = pd.DataFrame(doc_recored_remove)
    return df_ok, df_out


import networkx as nx
def get_max_connected_graph(edges_with_weight):
    G = nx.Graph()
    G.add_weighted_edges_from(edges_with_weight)
    connected_components = list(nx.connected_components(G))
    max_component = max(connected_components, key=len)
    max_subgraph = G.subgraph(max_component)
    
    # 计算最小生成树
    mst = nx.minimum_spanning_tree(max_subgraph)

    connected_graph = []
    for edge in mst.edges(data=False):
        connected_graph.append(edge)
    return connected_graph



def get_tree_similarity(struct_1, struct_2):
    
    connected_edges_1 = query_distance(struct_1,struct_1)
    if connected_edges_1:
        connected_edges_1 = get_max_connected_graph([(x,y,v) for x,y,v in connected_edges_1])

    connected_edges_2 = query_distance(struct_2,struct_2)
    if connected_edges_2:
        connected_edges_2 = get_max_connected_graph([(x,y,v) for x,y,v in connected_edges_2])
        
    if not connected_edges_2 or not connected_edges_1:
        return 0
    
    session = requests.Session()
    url = 'http://192.168.1.227:10004/wiki_api/api/tree/editDistance'
    # flag = 0, 只考虑结构
    graph_1 = [{'title':'a','edge':connected_edges_1}]
    graph_2 = [{'title':'b','edge':connected_edges_2}]
    struct_req_data = {
        "stra": json.dumps(graph_1),
        "strb": json.dumps(graph_2),
        "flag": 0
    }
    try:
        rep = session.post(url, json=struct_req_data, timeout=1)
    except requests.exceptions.ReadTimeout as e:
        print(struct_req_data)
        raise e
    tree_edit_distance = rep.json().get("data",{}).get('a_b')
    return 1- tree_edit_distance/255
    
