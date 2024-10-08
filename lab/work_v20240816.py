
import argparse
import pymongo
import tqdm
import psycopg
import random
from multiprocessing import Process, Queue

def get_collection():
    DATABASE = pymongo.MongoClient("192.168.1.222").temporary_token_similarity
    collection_node = DATABASE.token_similarity_node_v20240815
    collection_edge = DATABASE.token_similarity_edge_v20240815
    return collection_node,collection_edge

collection_node,collection_edge = get_collection()

pg_uri = os.environ.get('postgres_uri')

# 找到集合所有的边
def query_distance(tuple_set_1, tuple_set_2):
    ret = set()
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
                            ret.add(result)
                if tuple_set_1 != tuple_set_2:
                    for x in tuple_set_2:
                        greater_than_or_equal_x = list(item for item in tuple_set_1 if item > x)
                        if greater_than_or_equal_x:
                            cursor.execute("SELECT source_id,target_id,weight FROM edges WHERE source_id = %s AND target_id = ANY(%s);", 
                                        (x, greater_than_or_equal_x,)
                                        ) 
                            for result in cursor:
                                ret.add(result)
                else:
                    pass
                    # print("dup set")
            except TypeError as e:
                print(tuple_set_1)
                print(tuple_set_2)
                raise e
    return ret

def graph_link_maker():
    
    def process_worker(q_in):
        _,collection_edge = get_collection()
        while True:
            obj = q_in.get()
            if obj is None:
                q_in.put(None)
                break
            else:
                doc_id, sToken, tToken = obj
                edges = [(a,b,weight) for a,b,weight in query_distance(sToken-tToken,tToken-sToken)]
                collection_edge.update_one(
                    {'_id': doc_id},
                    {'$set': {'graph_link': edges}}
                )
           
    q_in = Queue(1_000_000)
    ps = []
    for _ in range(10):
        p = Process(target=process_worker, args=(q_in,))
        p.start()
        ps.append(p)
        
    for doc in tqdm.tqdm(collection_edge.find({'graph_link': {'$exists': False}, })):
        sID = doc['s']
        tID = doc['t']
        sdoc = collection_node.find_one({'_id': sID})
        tdoc = collection_node.find_one({'_id': tID})
        if not sdoc.get('token') or not tdoc.get('token'):
            print("edge disable", doc)
            continue
        sToken = set(sdoc.get('token'))
        tToken = set(tdoc.get('token'))
        if not sToken or not tToken:
            print("edge disable", doc)
            continue
    
        q_in.put((doc['_id'], sToken,tToken))

    q_in.put(None)
    for p in ps:
        p.join()


# sts 计算    
def task_1_sts_anlysis():
    from scipy.spatial.distance import (
        cosine,
        euclidean,
        minkowski,
        cityblock,
        chebyshev,
    )
    
    from modelscope import AutoModel
    JinNaAI_Model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method

    doc_list = [doc for doc in tqdm.tqdm(collection_edge.find({'sts_v1': {'$exists': False}, }))]
    for doc in tqdm.tqdm(doc_list):
        sID = doc['s']
        tID = doc['t']
        sdoc = collection_node.find_one({'_id': sID})
        tdoc = collection_node.find_one({'_id': tID})
        splaintext = sdoc.get('plaintext')
        tplaintext = tdoc.get('plaintext')
        
        if len(tplaintext) < 20 or len(splaintext) < 20:
            print("length too short",sID,tID)
            continue
        
        
        embeddings = JinNaAI_Model.encode([splaintext, tplaintext])
        cosine_similarity = cosine(embeddings[0],embeddings[1])
        euclidean_similarity = euclidean(embeddings[0],embeddings[1])
        minkowski_similarity = minkowski(embeddings[0],embeddings[1])
        cityblock_similarity = cityblock(embeddings[0],embeddings[1])
        chebyshev_similarity = chebyshev(embeddings[0],embeddings[1])
        collection_edge.update_one({'_id': doc['_id']},{"$set":{
            'sts_v1':{
                'cosine': cosine_similarity.item(),
                'euclidean': euclidean_similarity,
                'minkowski': minkowski_similarity,
                'cityblock': cityblock_similarity.item(),
                'chebyshev': chebyshev_similarity.item(),
            }
        }})

# sts 计算    
def task_2_token_anlysis():

    import pandas as pd
    import collections
    
    # doc_list = [doc for doc in tqdm.tqdm(collection_edge.find({'token_similarity_v1': {'$exists': False}, }))]
    # for doc in tqdm.tqdm(doc_list):
    for doc in  tqdm.tqdm(collection_edge.find({'token_similarity_v1': {'$exists': False}, })):
        sID = doc['s']
        tID = doc['t']
        sdoc = collection_node.find_one({'_id': sID})
        tdoc = collection_node.find_one({'_id': tID})
        stoken = set(sdoc.get('token',[]))
        ttoken = set(tdoc.get('token',[]))
        graph_link = doc.get('graph_link')
        
        if not stoken or not ttoken or not graph_link:
            print("link miss",sID,tID)
            continue
        
        
        dup_map_a = collections.defaultdict(list)
        dup_map_b = collections.defaultdict(list)

        for result in graph_link:
            dup_map_a[result[1]].append((result[2], result[0], result[1]))
            dup_map_b[result[0]].append((result[2], result[0], result[1]))
   
        doc_recored_ok = []    
        for item in (stoken & ttoken):
            doc_recored_ok.append(
                {
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
                'similarity': max(1-item[0],0 ),
            })

        df_ok = pd.DataFrame(doc_recored_ok)

        collection_edge.update_one({'_id': doc['_id']},{"$set":{
            'token_similarity_v1':df_ok['similarity'].mean()

        }})

def task_3_node_token_distance():
    doc_list= [doc for doc in  tqdm.tqdm(collection_node.find({'token': {'$exists': True},'token_distance': {'$exists': False}, }))]
    
    for doc in  tqdm.tqdm(doc_list):
        ret = query_distance(set([doc['_id']]), doc['token'])
        collection_node.update_one({'_id': doc['_id']},{"$set":{
            'token_distance':list(ret)
        }})
        # break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="this is help")
    parser.add_argument('method', type=int, 
                        choices=[-1, 0,1,2,3,4,5,6,7,8,9], 
                        help="this is help")

    args = parser.parse_args()

    # if args.method == -1:
    #     test()
    if args.method == 0:
        graph_link_maker()
    elif args.method == 1:
        task_1_sts_anlysis()
    elif args.method == 2:
        task_2_token_anlysis()
    elif args.method == 3:
        task_3_node_token_distance()